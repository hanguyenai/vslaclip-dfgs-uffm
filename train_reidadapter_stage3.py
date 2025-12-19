"""
Stage 3 Training Entry Script with DFGS Sampler.

Based on the paper:
"CLIP-DFGS: A Hard Sample Mining Method for CLIP in Generalizable Person Re-Identification"
https://arxiv.org/pdf/2410.11255v1
"""

from utils.logger import setup_logger
from datasets.make_video_dataloader import make_dataloader
from datasets import data_manager
from model.make_model_clipvideoreid_reidadapter_pbp import make_model
from solver.make_optimizer_prompt import make_optimizer_stage3
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss_video import make_loss
from processor.processor_videoreid_stage3 import do_train_stage3

import random
import torch
import numpy as np
import os
import argparse
from config import cfg


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser(description="Stage 3 ReID Training with DFGS Sampler")
    parser.add_argument(
        "--config_file", 
        default="configs/adapter/vit_adapter.yml", 
        help="path to config file", 
        type=str
    )
    parser.add_argument(
        "--stage2_weight", 
        required=True,
        help="path to Stage 2 checkpoint (required)", 
        type=str
    )
    parser.add_argument(
        "--dfgs_mode",
        default="text",
        choices=["text", "image"],
        help="DFGS mode: 'text' for DFGS_T(.) or 'image' for DFGS_I(.)",
        type=str
    )
    parser.add_argument(
        "opts", 
        help="Modify config options using the command-line", 
        default=None,
        nargs=argparse.REMAINDER
    )
    parser.add_argument("--local_rank", default=0, type=int)
    
    args = parser.parse_args()
    
    # Load config
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    
    cfg.merge_from_list(args.opts)
    
    # Add Stage 3 defaults if not present in config
    if not hasattr(cfg.SOLVER, 'STAGE3'):
        from yacs.config import CfgNode as CN
        cfg.SOLVER.STAGE3 = CN()
        cfg.SOLVER.STAGE3.IMS_PER_BATCH = cfg.SOLVER.STAGE2.IMS_PER_BATCH
        cfg.SOLVER.STAGE3.OPTIMIZER_NAME = "Adam"
        cfg.SOLVER.STAGE3.MAX_EPOCHS = 30
        cfg.SOLVER.STAGE3.BASE_LR = cfg.SOLVER.STAGE2.BASE_LR * 0.1  # 10x smaller
        cfg.SOLVER.STAGE3.WARMUP_METHOD = 'linear'
        cfg.SOLVER.STAGE3.WARMUP_ITERS = 10
        cfg.SOLVER.STAGE3.WARMUP_FACTOR = 0.1
        cfg.SOLVER.STAGE3.WEIGHT_DECAY = cfg.SOLVER.STAGE2.WEIGHT_DECAY
        cfg.SOLVER.STAGE3.WEIGHT_DECAY_BIAS = cfg.SOLVER.STAGE2.WEIGHT_DECAY_BIAS
        cfg.SOLVER.STAGE3.GAMMA = 0.1
        cfg.SOLVER.STAGE3.STEPS = (15, 25)
        cfg.SOLVER.STAGE3.CHECKPOINT_PERIOD = 5
        cfg.SOLVER.STAGE3.LOG_PERIOD = 50
        cfg.SOLVER.STAGE3.EVAL_PERIOD = 5
        # DFGS-specific parameters
        cfg.SOLVER.STAGE3.DFGS_K_NEIGHBORS = 10
        cfg.SOLVER.STAGE3.DFGS_M_DIFFICULTY = 2
        cfg.SOLVER.STAGE3.DFGS_SHUFFLE = True
        cfg.SOLVER.STAGE3.DFGS_UPDATE_EVERY = 5  # For DFGS_I(.)
    
    cfg.freeze()
    
    set_seed(cfg.SOLVER.SEED)
    
    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)
    
    # Setup output directory
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Setup logger
    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Stage 3 Training with DFGS Sampler")
    logger.info("Saving model in the path: {}".format(cfg.OUTPUT_DIR))
    logger.info(args)
    
    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    
    # Initialize distributed training if needed
    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    
    # Load dataset info (we need the raw dataset for DFGS sampler construction)
    logger.info("Loading dataset...")
    subset = getattr(cfg.DATASETS, 'SUBSET', 'case1_aerial_to_ground')
    dataset = data_manager.init_dataset(
        name=cfg.DATASETS.NAMES, 
        root=cfg.DATASETS.ROOT_DIR, 
        subset=subset
    )
    
    # Also load standard dataloaders for query/gallery
    (train_loader_stage2, train_loader_stage1,
     query_loader, gallery_loader,
     _, _,
     num_classes, num_query, num_camera) = make_dataloader(cfg)
    
    logger.info(f"Dataset loaded: {num_classes} identities, {num_camera} cameras")
    
    # Create model
    logger.info("Creating model...")
    model = make_model(cfg, num_class=num_classes, camera_num=num_camera, view_num=0)
    
    # Load Stage 2 checkpoint
    logger.info(f"Loading Stage 2 checkpoint from: {args.stage2_weight}")
    if not os.path.exists(args.stage2_weight):
        raise FileNotFoundError(f"Stage 2 checkpoint not found: {args.stage2_weight}")
    
    model.load_param_finetune(args.stage2_weight)
    logger.info("Stage 2 checkpoint loaded successfully")
    
    # Create loss function
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    
    # Create Stage 3 optimizer (10x smaller learning rate than Stage 2)
    optimizer_3stage, optimizer_center_3stage = make_optimizer_stage3(cfg, model, center_criterion)
    
    # Create Stage 3 scheduler
    scheduler_3stage = WarmupMultiStepLR(
        optimizer_3stage, 
        cfg.SOLVER.STAGE3.STEPS, 
        cfg.SOLVER.STAGE3.GAMMA, 
        cfg.SOLVER.STAGE3.WARMUP_FACTOR,
        cfg.SOLVER.STAGE3.WARMUP_ITERS, 
        cfg.SOLVER.STAGE3.WARMUP_METHOD
    )
    
    # Determine DFGS mode
    use_text_features = (args.dfgs_mode == "text")
    logger.info(f"Using DFGS mode: {'DFGS_T(.)' if use_text_features else 'DFGS_I(.)'}")
    
    # Run Stage 3 training
    logger.info("Starting Stage 3 training...")
    do_train_stage3(
        cfg,
        model,
        center_criterion,
        dataset.train,  # Raw dataset for DFGS sampler
        query_loader,
        gallery_loader,
        optimizer_3stage,
        optimizer_center_3stage,
        scheduler_3stage,
        loss_func,
        num_query,
        num_classes,
        args.local_rank,
        use_text_features=use_text_features
    )
    
    logger.info("Stage 3 training completed!")


if __name__ == '__main__':
    main()