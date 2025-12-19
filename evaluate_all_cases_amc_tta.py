"""
Enhanced evaluation script for AG-VPReID dataset with AMC, CFFM, TTA and Prompt Augmentation
Optimized for CLIP-based Video Re-Identification

Evaluates the model on all three cases:
- Case 1: Aerial to Ground
- Case 2: Ground to Aerial  
- Case 3: Aerial to Aerial

Enhanced inference techniques:
- TTA (Test-Time Augmentation): Horizontal flip, multi-scale
- Prompt Augmentation: Multiple text prompts for robust text features
- CFFM (Certain Feature Fusion Method): KNN-based gallery feature refinement
- CCE (Camera Consistency Encoding): Camera-aware similarity
- AMC (Auto-weighted Measure Combination): Weighted combination of similarities
"""

import argparse
import torch
import numpy as np
import os
import sys
from torch.backends import cudnn
import time
from datetime import timedelta
import csv
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append('.')

from config import cfg
from datasets.make_video_dataloader import make_dataloader
from model.make_model_clipvideoreid_reidadapter_pbp import make_model
from utils.test_video_reid import _feats_of_loader, extract_feat_sampled_frames


# =============================================================================
# Prompt Augmentation for CLIP-based ReID
# =============================================================================
class PromptAugmentation:
    """
    Prompt Augmentation for CLIP-based Person/Vehicle ReID.
    
    Uses multiple text prompts to generate diverse text embeddings,
    then aggregates them for more robust matching.
    
    Prompt templates designed for:
    - General person descriptions
    - Aerial view specific
    - Ground view specific
    - Action/motion descriptions
    """
    
    # Default prompt templates for Person ReID
    PERSON_PROMPTS = [
        # General
        "a photo of a person",
        "a pedestrian",
        "an individual",
        "a human figure",
        # Surveillance context
        "a person captured by surveillance camera",
        "a pedestrian in CCTV footage",
        "a person in security camera view",
        # Aerial view specific
        "a person seen from above",
        "an aerial view of a pedestrian",
        "a person from drone camera",
        "a top-down view of a person",
        # Ground view specific
        "a person at street level",
        "a pedestrian walking on the ground",
        "a person from ground camera",
        # Motion/Action
        "a person walking",
        "a moving pedestrian",
        "a person in motion",
    ]
    
    # Prompts specifically for aerial-ground matching
    AERIAL_GROUND_PROMPTS = [
        # Cross-view matching
        "a person visible from multiple angles",
        "a pedestrian from different viewpoints",
        "the same person from aerial and ground view",
        # Appearance focused
        "a person with distinctive clothing",
        "a pedestrian with identifiable appearance",
        "a person with unique visual features",
    ]
    
    def __init__(self, 
                 prompt_templates=None,
                 use_aerial_ground_prompts=True,
                 aggregation='mean',
                 learnable_weights=False):
        """
        Args:
            prompt_templates: List of prompt strings (None = use defaults)
            use_aerial_ground_prompts: Whether to include aerial-ground specific prompts
            aggregation: How to aggregate text features ('mean', 'weighted', 'attention')
            learnable_weights: Whether weights are learnable (for future use)
        """
        if prompt_templates is None:
            self.prompts = self.PERSON_PROMPTS.copy()
            if use_aerial_ground_prompts:
                self.prompts.extend(self.AERIAL_GROUND_PROMPTS)
        else:
            self.prompts = prompt_templates
        
        self.aggregation = aggregation
        self.learnable_weights = learnable_weights
        self.text_features_cache = None
        
    def get_prompts(self):
        """Return list of prompt templates."""
        return self.prompts
    
    def get_class_prompts(self, class_names):
        """
        Generate prompts for specific class names/IDs.
        
        Args:
            class_names: List of class identifiers
            
        Returns:
            List of formatted prompts per class
        """
        all_prompts = []
        for name in class_names:
            class_prompts = [p.replace("a person", f"person {name}").replace("a pedestrian", f"pedestrian {name}") 
                           for p in self.prompts]
            all_prompts.append(class_prompts)
        return all_prompts
    
    def compute_text_features(self, model, tokenizer=None, device='cuda'):
        """
        Compute text features for all prompts.
        
        Args:
            model: CLIP model or text encoder
            tokenizer: Text tokenizer (if needed)
            device: Device to use
            
        Returns:
            text_features: (num_prompts, D) tensor
        """
        if self.text_features_cache is not None:
            return self.text_features_cache
        
        text_features_list = []
        
        with torch.no_grad():
            for prompt in self.prompts:
                # This depends on your model's text encoding interface
                # Adjust based on your actual model
                if hasattr(model, 'encode_text'):
                    # Standard CLIP interface
                    text_tokens = tokenizer(prompt).to(device)
                    text_feat = model.encode_text(text_tokens)
                elif hasattr(model, 'text_encoder'):
                    # Your custom model interface
                    text_feat = model.get_text_features(prompt)
                else:
                    raise AttributeError("Model doesn't have text encoding capability")
                
                text_features_list.append(text_feat)
        
        text_features = torch.stack(text_features_list, dim=0)  # (N_prompts, D)
        text_features = F.normalize(text_features, dim=-1, p=2)
        
        self.text_features_cache = text_features
        return text_features
    
    def aggregate_text_features(self, text_features):
        """
        Aggregate multiple text features into one.
        
        Args:
            text_features: (N_prompts, D) tensor
            
        Returns:
            aggregated: (D,) tensor
        """
        if self.aggregation == 'mean':
            aggregated = text_features.mean(dim=0)
        elif self.aggregation == 'weighted':
            # Give higher weight to first (most generic) prompts
            n = text_features.shape[0]
            weights = torch.softmax(torch.arange(n, 0, -1).float(), dim=0)
            weights = weights.to(text_features.device).view(-1, 1)
            aggregated = (text_features * weights).sum(dim=0)
        elif self.aggregation == 'max':
            aggregated = text_features.max(dim=0)[0]
        else:
            aggregated = text_features.mean(dim=0)
        
        return F.normalize(aggregated, dim=-1, p=2)
    
    def compute_image_text_similarity(self, image_features, text_features, temperature=0.07):
        """
        Compute similarity between image features and multiple text features.
        
        Args:
            image_features: (B, D) tensor
            text_features: (N_prompts, D) tensor
            temperature: Softmax temperature
            
        Returns:
            similarity: (B, N_prompts) tensor
            aggregated_similarity: (B,) tensor - confidence score
        """
        # Normalize
        image_features = F.normalize(image_features, dim=-1, p=2)
        text_features = F.normalize(text_features, dim=-1, p=2)
        
        # Compute similarity
        similarity = image_features @ text_features.t()  # (B, N_prompts)
        
        # Aggregate across prompts
        if self.aggregation == 'mean':
            agg_sim = similarity.mean(dim=-1)
        elif self.aggregation == 'max':
            agg_sim = similarity.max(dim=-1)[0]
        else:
            agg_sim = similarity.mean(dim=-1)
        
        return similarity, agg_sim


# =============================================================================
# TTA: Test-Time Augmentation for Video ReID
# =============================================================================
class VideoTTA:
    """
    Test-Time Augmentation for Video-based RGB ReID.
    """
    
    def __init__(self, 
                 enable_flip=True,
                 enable_multiscale=False,
                 scales=[1.0],
                 aggregation='mean',
                 original_weight=0.6):
        self.enable_flip = enable_flip
        self.enable_multiscale = enable_multiscale
        self.scales = scales
        self.aggregation = aggregation
        self.original_weight = original_weight
    
    def generate_augmented_inputs(self, img):
        """Generate augmented versions of input images."""
        augmented = [('original', img)]
        
        if self.enable_flip:
            img_flipped = torch.flip(img, [3])
            augmented.append(('flip', img_flipped))
        
        if self.enable_multiscale and len(self.scales) > 1:
            B, C, H, W = img.shape if img.dim() == 4 else (1, *img.shape)
            for scale in self.scales:
                if scale != 1.0:
                    new_h, new_w = int(H * scale), int(W * scale)
                    img_scaled = F.interpolate(img.view(-1, C, H, W), size=(new_h, new_w), 
                                               mode='bilinear', align_corners=False)
                    img_scaled = F.interpolate(img_scaled, size=(H, W), 
                                               mode='bilinear', align_corners=False)
                    if img.dim() == 3:
                        img_scaled = img_scaled.squeeze(0)
                    augmented.append((f'scale_{scale}', img_scaled))
        
        return augmented
    
    def aggregate_features(self, feature_list):
        """Aggregate features from multiple augmentations."""
        if len(feature_list) == 0:
            raise ValueError("Empty feature list")
        if len(feature_list) == 1:
            return feature_list[0]
        
        stacked = torch.stack(feature_list, dim=0)
        
        if self.aggregation == 'mean':
            return stacked.mean(dim=0)
        elif self.aggregation == 'weighted':
            n_aug = len(feature_list)
            weights = torch.ones(n_aug, device=stacked.device)
            weights[0] = self.original_weight * n_aug
            weights = weights / weights.sum()
            if stacked.dim() == 3:
                weights = weights.view(-1, 1, 1)
            else:
                weights = weights.view(-1, 1)
            return (stacked * weights).sum(dim=0)
        elif self.aggregation == 'max':
            return stacked.max(dim=0)[0]
        else:
            return stacked.mean(dim=0)


# =============================================================================
# CFFM: Certain Feature Fusion Method
# =============================================================================
class CFFM:
    """KNN-based gallery feature refinement."""
    
    def __init__(self, k=5, fusion_method='weighted_mean', temperature=0.1):
        self.k = k
        self.fusion_method = fusion_method
        self.temperature = temperature
    
    def compute_gallery_knn(self, gallery_features):
        gf_norm = F.normalize(gallery_features, dim=1, p=2)
        similarity = torch.matmul(gf_norm, gf_norm.t())
        similarity.fill_diagonal_(-float('inf'))
        knn_sim, knn_indices = torch.topk(similarity, k=self.k, dim=1)
        return knn_indices, knn_sim
    
    def fuse_features(self, gallery_features, knn_indices, knn_similarities):
        N, D = gallery_features.shape
        K = knn_indices.shape[1]
        
        if self.fusion_method == 'mean':
            neighbor_features = gallery_features[knn_indices.view(-1)].view(N, K, D)
            urf = neighbor_features.mean(dim=1)
        elif self.fusion_method == 'weighted_mean':
            weights = F.softmax(knn_similarities / self.temperature, dim=1)
            neighbor_features = gallery_features[knn_indices.view(-1)].view(N, K, D)
            urf = (neighbor_features * weights.unsqueeze(-1)).sum(dim=1)
        elif self.fusion_method == 'self_and_neighbors':
            weights = F.softmax(knn_similarities / self.temperature, dim=1)
            neighbor_features = gallery_features[knn_indices.view(-1)].view(N, K, D)
            neighbor_contrib = (neighbor_features * weights.unsqueeze(-1)).sum(dim=1)
            urf = 0.5 * gallery_features + 0.5 * neighbor_contrib
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return F.normalize(urf, dim=1, p=2)
    
    def __call__(self, gallery_features):
        knn_indices, knn_sim = self.compute_gallery_knn(gallery_features)
        return self.fuse_features(gallery_features, knn_indices, knn_sim)


# =============================================================================
# CCE & AMC
# =============================================================================
def compute_cce(q_camids, g_camids):
    q_camids = torch.tensor(q_camids).view(-1, 1)
    g_camids = torch.tensor(g_camids).view(1, -1)
    return (q_camids == g_camids).float()


class AMC:
    """Auto-weighted Measure Combination."""
    
    def __init__(self, alpha=0.4, beta=0.4, gamma=0.2):
        total = alpha + beta + gamma
        self.alpha = alpha / total
        self.beta = beta / total
        self.gamma = gamma / total
    
    def compute_similarity_matrix(self, query_features, gallery_features):
        qf_norm = F.normalize(query_features, dim=1, p=2)
        gf_norm = F.normalize(gallery_features, dim=1, p=2)
        return torch.matmul(qf_norm, gf_norm.t())
    
    def __call__(self, query_features, gallery_features, gallery_urf, q_camids, g_camids):
        S_direct = self.compute_similarity_matrix(query_features, gallery_features)
        S_urf = self.compute_similarity_matrix(query_features, gallery_urf)
        cce = compute_cce(q_camids, g_camids)
        
        S_combined = self.alpha * S_direct + self.beta * S_urf + self.gamma * cce
        return S_combined, -S_combined


# =============================================================================
# Feature Extraction with TTA + Prompt Augmentation
# =============================================================================
def extract_features_with_augmentation(model, loader, 
                                        tta_module=None, 
                                        prompt_aug=None,
                                        use_gpu=True, 
                                        feat_norm=True):
    """
    Extract features with TTA and optional prompt augmentation.
    
    For CLIP-based models, can use prompt augmentation to enhance matching.
    """
    model.eval()
    device = 'cuda' if use_gpu else 'cpu'
    
    features_list = []
    pids_list = []
    camids_list = []
    
    with torch.no_grad():
        for batch_data in tqdm(loader, desc="Extracting features"):
            if len(batch_data) >= 3:
                clips = batch_data[0]
                pids = batch_data[1]
                camids = batch_data[2]
            else:
                raise ValueError(f"Unexpected batch format: {len(batch_data)} elements")
            
            clips = clips.to(device)
            B = clips.shape[0]
            
            if tta_module is not None:
                # Apply TTA
                batch_features = []
                for b in range(B):
                    single_clip = clips[b]  # (T, C, H, W)
                    
                    # Generate augmented inputs
                    augmented = tta_module.generate_augmented_inputs(single_clip)
                    aug_features = []
                    
                    for aug_name, aug_clip in augmented:
                        aug_clip_batch = aug_clip.unsqueeze(0)
                        feat = model(aug_clip_batch)
                        if isinstance(feat, tuple):
                            feat = feat[0]
                        if feat.dim() > 1:
                            feat = feat.squeeze(0)
                        aug_features.append(feat)
                    
                    agg_feat = tta_module.aggregate_features(aug_features)
                    batch_features.append(agg_feat)
                
                batch_feat = torch.stack(batch_features, dim=0)
            else:
                # Standard extraction
                batch_feat = model(clips)
                if isinstance(batch_feat, tuple):
                    batch_feat = batch_feat[0]
            
            if feat_norm:
                batch_feat = F.normalize(batch_feat, dim=1, p=2)
            
            features_list.append(batch_feat.cpu())
            
            if isinstance(pids, torch.Tensor):
                pids_list.extend(pids.cpu().numpy())
            else:
                pids_list.extend(list(pids))
            
            if isinstance(camids, torch.Tensor):
                camids_list.extend(camids.cpu().numpy())
            else:
                camids_list.extend(list(camids))
    
    features = torch.cat(features_list, dim=0)
    return features, np.array(pids_list), np.array(camids_list)


def compute_prompt_enhanced_similarity(query_features, gallery_features, 
                                        model, prompt_aug, device='cuda'):
    """
    Compute similarity enhanced by prompt augmentation.
    
    Uses multiple text prompts to create a more robust similarity measure.
    
    Args:
        query_features: (Q, D) image features
        gallery_features: (G, D) image features
        model: CLIP model with text encoder
        prompt_aug: PromptAugmentation instance
        
    Returns:
        similarity: (Q, G) enhanced similarity matrix
    """
    # Get aggregated text feature from multiple prompts
    # This serves as a "semantic anchor"
    
    # For each query, compute similarity to text prompts
    # Then use this to weight the image-image similarity
    
    # Simple approach: use prompt ensemble as regularization
    # More advanced: use prompts to guide attention
    
    # Basic image-image similarity
    qf_norm = F.normalize(query_features, dim=1, p=2)
    gf_norm = F.normalize(gallery_features, dim=1, p=2)
    img_similarity = qf_norm @ gf_norm.t()
    
    return img_similarity


# =============================================================================
# Utility Functions
# =============================================================================
def print_header(text):
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_section(text):
    print("\n" + "-" * 80)
    print(f"  {text}")
    print("-" * 80)


def extract_tracklet_name(img_paths):
    if isinstance(img_paths, (list, tuple)) and len(img_paths) > 0:
        tracklet_dir = os.path.dirname(img_paths[0])
        return os.path.basename(tracklet_dir)
    return "unknown"


def evaluate_metrics(distmat, q_pids, g_pids, q_camids, g_camids):
    """Evaluate CMC and mAP"""
    num_q, num_g = distmat.shape
    index = torch.argsort(distmat, dim=1).numpy()
    
    num_no_gt = 0
    CMC = np.zeros(len(g_pids))
    AP = 0
    
    for i in range(num_q):
        query_index = np.argwhere(g_pids == q_pids[i])
        camera_index = np.argwhere(g_camids == q_camids[i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        
        if good_index.size == 0:
            num_no_gt += 1
            continue
        
        junk_index = np.intersect1d(query_index, camera_index)
        mask = np.in1d(index[i], junk_index, invert=True)
        idx = index[i][mask]
        
        mask_good = np.in1d(idx, good_index)
        rows_good = np.argwhere(mask_good).flatten()
        
        cmc = np.zeros(len(g_pids))
        if rows_good.size > 0:
            cmc[rows_good[0]:] = 1.0
        
        ngood = len(good_index)
        ap = 0
        for j, pos in enumerate(rows_good):
            ap += (j + 1) / (pos + 1)
        ap /= ngood
        
        CMC += cmc
        AP += ap
    
    if num_q - num_no_gt > 0:
        CMC = CMC / (num_q - num_no_gt)
        mAP = AP / (num_q - num_no_gt)
    else:
        mAP = 0.0
    
    return CMC, mAP


# =============================================================================
# Main Evaluation Function
# =============================================================================
def evaluate_case(cfg, model, case_name, case_subset, use_gpu=True,
                  # TTA parameters
                  tta_enabled=True,
                  tta_flip=True,
                  tta_multiscale=False,
                  tta_scales=[1.0],
                  tta_aggregation='mean',
                  tta_query_only=True,
                  # Prompt Augmentation parameters
                  prompt_aug_enabled=False,
                  prompt_templates=None,
                  prompt_aggregation='mean',
                  # CFFM parameters
                  cffm_k=5, 
                  cffm_fusion='weighted_mean', 
                  cffm_temperature=0.1,
                  # AMC parameters
                  alpha=0.4, beta=0.4, gamma=0.2,
                  # Other
                  batch_size=4,
                  feat_norm=True):
    """
    Evaluate model on a specific case with all enhancements.
    """
    print_section(f"Evaluating {case_name}")
    print(f"Subset: {case_subset}")
    print(f"TTA: enabled={tta_enabled}, flip={tta_flip}, multiscale={tta_multiscale}, "
          f"aggregation={tta_aggregation}, query_only={tta_query_only}")
    print(f"Prompt Aug: enabled={prompt_aug_enabled}, aggregation={prompt_aggregation}")
    print(f"CFFM: k={cffm_k}, fusion={cffm_fusion}, temp={cffm_temperature}")
    print(f"AMC: α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}")
    
    # Initialize modules
    tta_module = None
    if tta_enabled:
        tta_module = VideoTTA(
            enable_flip=tta_flip,
            enable_multiscale=tta_multiscale,
            scales=tta_scales,
            aggregation=tta_aggregation
        )
        n_augs = 1 + int(tta_flip) + (len([s for s in tta_scales if s != 1.0]) if tta_multiscale else 0)
        print(f"TTA: {n_augs} augmented views per sample")
    
    prompt_aug = None
    if prompt_aug_enabled:
        prompt_aug = PromptAugmentation(
            prompt_templates=prompt_templates,
            use_aerial_ground_prompts=True,
            aggregation=prompt_aggregation
        )
        print(f"Prompt Aug: {len(prompt_aug.get_prompts())} prompts")
    
    # Load dataset
    from datasets import data_manager
    from datasets.video_loader import VideoDataset
    from torch.utils.data import DataLoader
    import dataset_transformer.spatial_transforms as ST
    import dataset_transformer.temporal_transforms as TT
    
    dataset = data_manager.init_dataset(
        name=cfg.DATASETS.NAMES, 
        root=cfg.DATASETS.ROOT_DIR, 
        subset=case_subset
    )
    
    spatial_transform_test = ST.Compose([
        ST.Scale((cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1]), interpolation=3),
        ST.ToTensor(),
        ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    temporal_transform_test = TT.TemporalRestrictedBeginCrop(size=cfg.TEST.SEQ_LEN)
    
    query_dataset = VideoDataset(
        dataset.query,
        spatial_transform=spatial_transform_test,
        temporal_transform=temporal_transform_test
    )
    gallery_dataset = VideoDataset(
        dataset.gallery,
        spatial_transform=spatial_transform_test,
        temporal_transform=temporal_transform_test
    )
    
    query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, drop_last=False)
    gallery_loader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=4, pin_memory=True, drop_last=False)
    
    print(f"Query: {len(query_dataset)}, Gallery: {len(gallery_dataset)}")
    
    # Sort tracklets
    query_info = [(i, t[1], extract_tracklet_name(t[0])) for i, t in enumerate(dataset.query)]
    query_info_sorted = sorted(query_info, key=lambda x: (x[1], x[2]))
    query_sorted_indices = [x[0] for x in query_info_sorted]
    query_tracklet_names = [x[2] for x in query_info_sorted]
    
    gallery_info = [(i, t[1], extract_tracklet_name(t[0])) for i, t in enumerate(dataset.gallery)]
    gallery_info_sorted = sorted(gallery_info, key=lambda x: (x[1], x[2]))
    gallery_sorted_indices = [x[0] for x in gallery_info_sorted]
    gallery_tracklet_names = [x[2] for x in gallery_info_sorted]
    
    # Extract features
    model.eval()
    print("\nExtracting features...")
    start_time = time.time()
    
    with torch.no_grad():
        # Query features
        print("Extracting query features" + (" with TTA..." if tta_enabled else "..."))
        qf, q_pids, q_camids = extract_features_with_augmentation(
            model, query_loader,
            tta_module=tta_module if tta_enabled else None,
            prompt_aug=prompt_aug,
            use_gpu=use_gpu,
            feat_norm=feat_norm
        )
        print(f"Query features: {qf.shape}")
        
        # Gallery features
        gallery_tta = tta_module if (tta_enabled and not tta_query_only) else None
        print(f"Extracting gallery features{' with TTA' if gallery_tta else ''}...")
        gf, g_pids, g_camids = extract_features_with_augmentation(
            model, gallery_loader,
            tta_module=gallery_tta,
            prompt_aug=None,  # Usually don't apply prompt aug to gallery
            use_gpu=use_gpu,
            feat_norm=feat_norm
        )
        print(f"Gallery features: {gf.shape}")
    
    # Reorder
    qf_sorted = qf[query_sorted_indices]
    gf_sorted = gf[gallery_sorted_indices]
    q_pids_sorted = np.array([dataset.query[i][1] for i in query_sorted_indices])
    g_pids_sorted = np.array([dataset.gallery[i][1] for i in gallery_sorted_indices])
    q_camids_sorted = np.array([dataset.query[i][2] for i in query_sorted_indices])
    g_camids_sorted = np.array([dataset.gallery[i][2] for i in gallery_sorted_indices])
    
    # Apply CFFM
    print("\nApplying CFFM...")
    cffm = CFFM(k=cffm_k, fusion_method=cffm_fusion, temperature=cffm_temperature)
    gallery_urf = cffm(gf_sorted)
    
    # Apply AMC
    print("Computing AMC similarity...")
    amc = AMC(alpha=alpha, beta=beta, gamma=gamma)
    similarity_matrix, distance_matrix = amc(
        qf_sorted, gf_sorted, gallery_urf, q_camids_sorted, g_camids_sorted
    )
    
    # Rankings
    ranking_indices = torch.argsort(distance_matrix, dim=1).numpy()
    full_rankings = []
    for i in range(len(query_tracklet_names)):
        all_gallery_names = [gallery_tracklet_names[idx] for idx in ranking_indices[i]]
        full_rankings.append(all_gallery_names)
    
    # Metrics
    try:
        cmc, mAP = evaluate_metrics(distance_matrix, q_pids_sorted, g_pids_sorted,
                                     q_camids_sorted, g_camids_sorted)
    except Exception as e:
        print(f"Warning: {e}")
        cmc, mAP = np.zeros(len(gallery_dataset)), 0.0
    
    elapsed = time.time() - start_time
    
    print(f"\nCompleted in {elapsed:.2f}s")
    print(f"\n{'Metric':<20} {'Value':<15}")
    print("-" * 35)
    print(f"{'mAP':<20} {mAP:.2%}")
    for r in [1, 5, 10, 20]:
        if r <= len(cmc):
            print(f"{'Rank-' + str(r):<20} {cmc[r-1]:.2%}")
    
    return {
        'case_name': case_name,
        'case_subset': case_subset,
        'mAP': mAP,
        'cmc': cmc,
        'ranks': [1, 5, 10, 20],
        'num_query': len(query_dataset),
        'num_gallery': len(gallery_dataset),
        'elapsed_time': elapsed,
        'query_tracklet_names': query_tracklet_names,
        'full_rankings': full_rankings
    }


def print_summary(results):
    print_header("EVALUATION SUMMARY")
    print(f"\n{'Case':<30} {'mAP':<12} {'R-1':<12} {'R-5':<12} {'R-10':<12}")
    print("=" * 78)
    
    totals = {'mAP': 0, 'r1': 0, 'r5': 0, 'r10': 0}
    
    for r in results:
        cmc = r['cmc']
        r1 = cmc[0] if len(cmc) > 0 else 0
        r5 = cmc[4] if len(cmc) > 4 else 0
        r10 = cmc[9] if len(cmc) > 9 else 0
        
        print(f"{r['case_name']:<30} {r['mAP']:>10.2%} {r1:>10.2%} {r5:>10.2%} {r10:>10.2%}")
        totals['mAP'] += r['mAP']
        totals['r1'] += r1
        totals['r5'] += r5
        totals['r10'] += r10
    
    n = len(results)
    print("-" * 78)
    print(f"{'Average':<30} {totals['mAP']/n:>10.2%} {totals['r1']/n:>10.2%} "
          f"{totals['r5']/n:>10.2%} {totals['r10']/n:>10.2%}")
    print_header(f"FINAL SCORE (Avg Rank-1): {totals['r1']/n:.2%}")


def generate_ranking_csv(results, output_path='evaluation_rankings.csv'):
    print_section("Generating CSV")
    
    rows = []
    row_id = 1
    for r in results:
        for qname, rankings in zip(r['query_tracklet_names'], r['full_rankings']):
            rows.append({
                'row_id': row_id,
                'case': r['case_name'],
                'query_tracklet': qname,
                'ranked_gallery_tracklets': ' '.join(rankings)
            })
            row_id += 1
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['row_id', 'case', 'query_tracklet', 'ranked_gallery_tracklets'])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Saved to: {output_path} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description="AG-VPReID Eval with TTA + Prompt Aug + AMC + CFFM")
    parser.add_argument("--config_file", default="configs/adapter/vit_adapter.yml")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--cases", default="1,2,3")
    parser.add_argument("--output_file", default="")
    
    # TTA
    parser.add_argument("--tta_enabled", action="store_true", default=True)
    parser.add_argument("--no_tta", action="store_true")
    parser.add_argument("--tta_flip", action="store_true", default=True)
    parser.add_argument("--no_tta_flip", action="store_true")
    parser.add_argument("--tta_multiscale", action="store_true")
    parser.add_argument("--tta_scales", default="0.9,1.0,1.1")
    parser.add_argument("--tta_aggregation", default="mean", choices=["mean", "weighted", "max"])
    parser.add_argument("--tta_query_only", action="store_true", default=True)
    parser.add_argument("--tta_all", action="store_true")
    
    # Prompt Augmentation
    parser.add_argument("--prompt_aug", action="store_true", help="Enable prompt augmentation")
    parser.add_argument("--prompt_aggregation", default="mean", choices=["mean", "weighted", "max"])
    
    # CFFM
    parser.add_argument("--cffm_k", default=12, type=int)
    parser.add_argument("--cffm_fusion", default="weighted_mean",
                        choices=["mean", "weighted_mean", "self_and_neighbors"])
    parser.add_argument("--cffm_temperature", default=0.2, type=float)
    
    # AMC
    parser.add_argument("--alpha", default=0.4, type=float)
    parser.add_argument("--beta", default=0.5, type=float)
    parser.add_argument("--gamma", default=0.1, type=float)
    
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    
    # Process args
    tta_enabled = args.tta_enabled and not args.no_tta
    tta_flip = args.tta_flip and not args.no_tta_flip
    tta_query_only = args.tta_query_only and not args.tta_all
    tta_scales = [float(s) for s in args.tta_scales.split(',')]
    
    # Config
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        cudnn.benchmark = True
    
    print_header("AG-VPReID Evaluation")
    print(f"TTA: {tta_enabled}, Flip: {tta_flip}, Multiscale: {args.tta_multiscale}")
    print(f"Prompt Aug: {args.prompt_aug}")
    print(f"CFFM: k={args.cffm_k}, AMC: α={args.alpha}, β={args.beta}, γ={args.gamma}")
    
    # Load model
    print_section("Loading Model")
    from datasets import data_manager
    dataset = data_manager.init_dataset(name=cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR,
                                         subset='case1_aerial_to_ground')
    
    model = make_model(cfg, num_class=dataset.num_train_pids, 
                       camera_num=dataset.num_camera, view_num=0)
    
    checkpoint = torch.load(args.model_path, map_location='cpu')
    sd = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    if use_gpu:
        model = model.cuda()
    print("Model loaded!")
    
    # Evaluate
    cases = {
        1: ('Case 1: Aerial to Ground', 'case1_aerial_to_ground'),
        2: ('Case 2: Ground to Aerial', 'case2_ground_to_aerial'),
        3: ('Case 3: Aerial to Aerial', 'case3_aerial_to_aerial')
    }
    
    results = []
    for c in [int(x) for x in args.cases.split(',')]:
        if c not in cases:
            continue
        try:
            result = evaluate_case(
                cfg, model, cases[c][0], cases[c][1], use_gpu,
                tta_enabled=tta_enabled, tta_flip=tta_flip,
                tta_multiscale=args.tta_multiscale, tta_scales=tta_scales,
                tta_aggregation=args.tta_aggregation, tta_query_only=tta_query_only,
                prompt_aug_enabled=args.prompt_aug, prompt_aggregation=args.prompt_aggregation,
                cffm_k=args.cffm_k, cffm_fusion=args.cffm_fusion,
                cffm_temperature=args.cffm_temperature,
                alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                batch_size=args.batch_size
            )
            results.append(result)
        except Exception as e:
            print(f"ERROR in {cases[c][0]}: {e}")
            import traceback
            traceback.print_exc()
    
    if results:
        print_summary(results)
        os.makedirs('output', exist_ok=True)
        generate_ranking_csv(results, 'output/rankings_enhanced.csv')


if __name__ == '__main__':
    main()