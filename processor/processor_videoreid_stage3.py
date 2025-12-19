"""
Stage 3 Processor for Video ReID with DFGS Sampler.

This stage fine-tunes the model trained in Stage 2 using the Depth-First Graph Sampler
for hard sample mining, following the CLIP-DFGS paper methodology.
"""

import logging
import os
import time
from datetime import timedelta
from collections import defaultdict

import torch
import torch.nn as nn
from torch.cuda import amp
import torch.distributed as dist
from torch.utils.data import DataLoader
import numpy as np

from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from loss.supcontrast import SupConLoss
from utils.test_video_reid import test, _eval_format_logger

from datasets.dfgs_sampler import DFGSSamplerWithGraph, build_id_to_indices, compute_pairwise_distances
from datasets.video_loader import VideoDataset
from dataset_transformer import temporal_transforms as TT, spatial_transforms as ST


def compute_text_features_for_all_pids(model, num_classes, batch_size, device):
    """
    Compute text features for all person IDs.
    Used for DFGS_T(.) variant where graph is based on text encoder features.
    """
    text_features = []
    i_ter = num_classes // batch_size
    left = num_classes - batch_size * (num_classes // batch_size)
    if left != 0:
        i_ter = i_ter + 1
    
    with torch.no_grad():
        for i in range(i_ter):
            if i + 1 != i_ter:
                l_list = torch.arange(i * batch_size, (i + 1) * batch_size)
            else:
                l_list = torch.arange(i * batch_size, num_classes)
            with amp.autocast(enabled=True):
                text_feature = model(label=l_list, get_text=True)
            text_features.append(text_feature.cpu())
    
    text_features = torch.cat(text_features, 0)
    return text_features


def compute_image_features_for_pids(model, train_loader, device):
    """
    Compute average image features for each person ID.
    Used for DFGS_I(.) variant where graph is based on image encoder features.
    """
    pid_features = defaultdict(list)
    
    model.eval()
    with torch.no_grad():
        for batch_data in train_loader:
            if len(batch_data) == 3:
                img, vid, _ = batch_data
            else:
                img, vid, _, _, _, _, _, _ = batch_data
            
            img = img.to(device)
            batch_size = img.shape[0]
            
            with amp.autocast(enabled=True):
                image_feature = model(img, vid.to(device), get_image=True)
            
            for i, (pid, feat) in enumerate(zip(vid, image_feature)):
                pid_features[pid.item()].append(feat.cpu())
    
    # Average features per pid
    pid_list = sorted(pid_features.keys())
    avg_features = []
    for pid in pid_list:
        feats = torch.stack(pid_features[pid], dim=0)
        avg_feat = feats.mean(dim=0)
        avg_features.append(avg_feat)
    
    avg_features = torch.stack(avg_features, dim=0)
    return avg_features, pid_list


def build_dfgs_dataloader(cfg, dataset_train, num_classes, model, device, use_text_features=True):
    """
    Build DataLoader with DFGS Sampler.
    
    Args:
        cfg: Configuration object
        dataset_train: Training dataset (list of tracklets)
        num_classes: Number of person IDs
        model: The model (for feature extraction)
        device: CUDA device
        use_text_features: If True, use DFGS_T(.) (text features for graph)
                          If False, use DFGS_I(.) (image features for graph)
    
    Returns:
        DataLoader with DFGS sampler, sampler reference for updates
    """
    # Build spatial and temporal transforms
    spatial_transform = ST.Compose([
        ST.Scale(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        ST.RandomHorizontalFlip(cfg.INPUT.PROB),
        ST.ToTensor(),
        ST.Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD),
        ST.RandomErasing(probability=cfg.INPUT.RE_PROB)
    ])
    temporal_transform = TT.TemporalRestrictedCrop(size=cfg.DATALOADER.SEQ_LEN)
    
    # Create dataset
    train_dataset = VideoDataset(
        dataset_train,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform
    )
    
    # Build id_to_indices mapping
    id_to_indices = build_id_to_indices(dataset_train)
    pid_list = sorted(id_to_indices.keys())
    
    # Compute pairwise distances
    print("Computing features for DFGS graph construction...")
    if use_text_features:
        # DFGS_T(.): Use text encoder features
        batch_size = cfg.SOLVER.STAGE3.IMS_PER_BATCH
        features = compute_text_features_for_all_pids(model, num_classes, batch_size, device)
        # pid_list for text features is just 0 to num_classes-1
        pid_list = list(range(num_classes))
    else:
        # DFGS_I(.): Use image encoder features
        # Create a simple loader for feature extraction
        temp_loader = DataLoader(
            train_dataset,
            batch_size=cfg.SOLVER.STAGE3.IMS_PER_BATCH,
            shuffle=False,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=True
        )
        features, pid_list = compute_image_features_for_pids(model, temp_loader, device)
    
    # Compute pairwise distances
    print("Computing pairwise distances...")
    pairwise_distances = compute_pairwise_distances(features, metric='euclidean')
    
    # Get DFGS parameters from config
    P = cfg.SOLVER.STAGE3.IMS_PER_BATCH // cfg.DATALOADER.NUM_INSTANCE
    K = cfg.DATALOADER.NUM_INSTANCE
    k_neighbors = getattr(cfg.SOLVER.STAGE3, 'DFGS_K_NEIGHBORS', 10)
    m_difficulty = getattr(cfg.SOLVER.STAGE3, 'DFGS_M_DIFFICULTY', 2)
    shuffle_graph = getattr(cfg.SOLVER.STAGE3, 'DFGS_SHUFFLE', True)
    
    print(f"DFGS Parameters: P={P}, K={K}, k_neighbors={k_neighbors}, m_difficulty={m_difficulty}")
    
    # Create DFGS sampler
    dfgs_sampler = DFGSSamplerWithGraph(
        data_source=dataset_train,
        pairwise_distances=pairwise_distances,
        pid_list=pid_list,
        P=P,
        K=K,
        k_neighbors=k_neighbors,
        m_difficulty=m_difficulty,
        shuffle_graph=shuffle_graph
    )
    
    # Create DataLoader with DFGS sampler
    train_loader = DataLoader(
        train_dataset,
        sampler=dfgs_sampler,
        batch_size=cfg.SOLVER.STAGE3.IMS_PER_BATCH,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    return train_loader, dfgs_sampler


def do_train_stage3(cfg,
                    model,
                    center_criterion,
                    dataset_train,
                    query_loader,
                    gallery_loader,
                    optimizer,
                    optimizer_center,
                    scheduler,
                    loss_fn,
                    num_query,
                    num_classes,
                    local_rank,
                    use_text_features=True):
    """
    Stage 3 training with DFGS sampler.
    
    This function performs fine-tuning using hard samples mined by the
    Depth-First Graph Sampler following the CLIP-DFGS methodology.
    """
    log_period = cfg.SOLVER.STAGE3.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE3.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE3.EVAL_PERIOD
    
    device = "cuda"
    epochs = cfg.SOLVER.STAGE3.MAX_EPOCHS
    
    logger = logging.getLogger("transreid.train")
    logger.info('Starting Stage 3 training with DFGS sampler')
    
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
            num_classes_model = model.module.num_classes
        else:
            num_classes_model = model.num_classes
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    
    # Build DFGS DataLoader
    logger.info("Building DFGS DataLoader...")
    train_loader_stage3, dfgs_sampler = build_dfgs_dataloader(
        cfg, dataset_train, num_classes, model, device, use_text_features
    )
    
    # Collect all text features for i2t loss computation
    batch = cfg.SOLVER.STAGE3.IMS_PER_BATCH
    i_ter = num_classes // batch
    left = num_classes - batch * (num_classes // batch)
    if left != 0:
        i_ter = i_ter + 1
    
    text_features = []
    print("Collecting text features for all IDs...")
    with torch.no_grad():
        for i in range(i_ter):
            if i + 1 != i_ter:
                l_list = torch.arange(i * batch, (i + 1) * batch)
            else:
                l_list = torch.arange(i * batch, num_classes)
            with amp.autocast(enabled=True):
                text_feature = model(label=l_list, get_text=True)
            text_features.append(text_feature.cpu())
        text_features = torch.cat(text_features, 0).cuda()
    
    print("Text features collected. Starting Stage 3 training!")
    
    all_start_time = time.monotonic()
    best_rank_1 = 0.0
    best_mAP = 0.0
    
    # Update graph frequency for DFGS_I(.) - update every N epochs
    update_graph_every = getattr(cfg.SOLVER.STAGE3, 'DFGS_UPDATE_EVERY', 5)
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        
        scheduler.step()
        
        # Optionally update DFGS graph for DFGS_I(.) variant
        if not use_text_features and epoch > 1 and epoch % update_graph_every == 0:
            logger.info(f"Updating DFGS graph at epoch {epoch}...")
            # Recompute image features and update sampler
            temp_loader = DataLoader(
                train_loader_stage3.dataset,
                batch_size=cfg.SOLVER.STAGE3.IMS_PER_BATCH,
                shuffle=False,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                pin_memory=True
            )
            model.eval()
            features, pid_list = compute_image_features_for_pids(model, temp_loader, device)
            new_distances = compute_pairwise_distances(features, metric='euclidean')
            dfgs_sampler.update_distances(new_distances)
        
        model.train()
        
        for n_iter, batch_data in enumerate(train_loader_stage3):
            # Handle both original and extended dataset formats
            if len(batch_data) == 3:
                img, vid, target_cam = batch_data
                altitude = distance = angle = None
            else:
                img, vid, target_cam, altitude, distance, angle, aerial_distance, point_id = batch_data
            
            target_view = target_cam
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            
            img = img.to(device)
            target = vid.to(device)
            
            # Prepare metadata for model if available
            if altitude is not None:
                altitude = altitude.to(device) if isinstance(altitude, torch.Tensor) else torch.tensor(altitude).to(device)
                distance = distance.to(device) if isinstance(distance, torch.Tensor) else torch.tensor(distance).to(device)
                angle = angle.to(device) if isinstance(angle, torch.Tensor) else torch.tensor(angle).to(device)
            
            if cfg.MODEL.PBP_CAMERA:
                target_cam = target_cam.to(device)
            else:
                target_cam = None
            
            if cfg.MODEL.PBP_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None
            
            with amp.autocast(enabled=True):
                score, feat, image_features = model(
                    x=img,
                    label=target,
                    cam_label=target_cam,
                    view_label=target_view
                )
                text_feature = model(label=target, get_text=True)
                loss_i2t = xent(image_features, text_feature, target, target)
                loss_t2i = xent(text_feature, image_features, target, target)
                logits = image_features @ text_features.t()
                loss = loss_fn(score, feat, target, target_cam, logits, loss_t2i, loss_i2t)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            
            acc = (logits.max(1)[1] == target).float().mean()
            
            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)
            
            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                           .format(epoch, (n_iter + 1), len(train_loader_stage3),
                                   loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))
        
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                       .format(epoch, time_per_batch, 
                               train_loader_stage3.batch_size / time_per_batch))
        
        # Save checkpoint
        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                              os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage3_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                          os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage3_{}.pth'.format(epoch)))
        
        # Evaluation (commented out for anonymized test sets)
        # Uncomment below to enable evaluation with proper ground truth
        # if epoch % eval_period == 0:
        #     if not cfg.MODEL.DIST_TRAIN:
        #         use_gpu = True
        #         cmc, mAP, ranks = test(model, query_loader, gallery_loader, use_gpu, cfg)
        #         ptr = "mAP: {:.2%}".format(mAP)
        #         if cmc[0] > best_rank_1:
        #             best_rank_1 = cmc[0]
        #             torch.save(model.state_dict(),
        #                       os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage3_rank1_best.pth'))
        #         if mAP > best_mAP:
        #             best_mAP = mAP
        #             torch.save(model.state_dict(),
        #                       os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage3_mAP_best.pth'))
        #         for r in ranks:
        #             ptr += " | R-{:<3}: {:.2%}".format(r, cmc[r - 1])
        #         logger.info(ptr)
    
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage 3 total running time: {}".format(total_time))
    
    # Save final model
    torch.save(model.state_dict(),
              os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage3_final.pth'))
    logger.info("Stage 3 training completed. Model saved to: {}".format(cfg.OUTPUT_DIR))