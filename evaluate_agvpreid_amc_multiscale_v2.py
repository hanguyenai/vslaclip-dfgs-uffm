"""
Comprehensive evaluation script for AG-VPReID dataset with AMC, UFFM, and Advanced Multiscale
FIXED VERSION: Compatible with ViT models that have fixed positional embeddings

Key fix: All processing outputs FIXED SIZE (256x128) to match ViT positional embeddings
Multi-scale is achieved by:
1. Crop/zoom at different scales BEFORE resizing to fixed size
2. Different aspect ratio handling BEFORE final resize
3. All augmentations output the same final size

Optimized for Aerial-Ground Person Re-identification challenges:
- Aspect ratio preservation (aerial top-down vs ground front view)
- Low resolution enhancement  
- Multi-scale processing with proper handling for ViT
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
import math

sys.path.append('.')

from config import cfg
from datasets.make_video_dataloader import make_dataloader
from model.make_model_clipvideoreid_reidadapter_pbp import make_model
from utils.test_video_reid import _feats_of_loader, extract_feat_sampled_frames


# =============================================================================
# ADVANCED MULTISCALE: Fixed for ViT (always outputs base_size)
# =============================================================================
class AdvancedMultiscaleExtractor:
    """
    Advanced Multi-scale Feature Extraction for Aerial-Ground Person ReID
    
    IMPORTANT: This version is compatible with ViT models that have fixed positional embeddings.
    All processing outputs the SAME FIXED SIZE (e.g., 256x128) regardless of scale.
    
    Multi-scale is achieved by:
    1. Scale-aware cropping: Crop center region at different scales before resize
    2. Aspect ratio aware processing: Handle different AR before final resize
    3. Resolution enhancement: Upscale low-res before processing
    
    All augmentations → Fixed output size → Model
    """
    
    def __init__(self, 
                 base_size=(256, 128),
                 # Multi-scale via cropping (all outputs same size)
                 scales=[1.0],  # 1.0 = full image, 0.8 = center 80%, 1.2 = zoom out (pad)
                 use_flip=True,
                 # Aspect ratio handling
                 preserve_aspect_ratio=True,
                 padding_mode='constant',  # 'constant', 'reflect', 'replicate'
                 padding_value=0,
                 # Resolution enhancement for low-res
                 min_resolution=64,
                 upscale_factor=2.0,
                 # Additional square crop for aerial (outputs same size)
                 use_square_crop=True,
                 # Fusion
                 fusion_method='weighted'):
        """
        Args:
            base_size: FIXED output size (H, W) - must match model's expected input
            scales: Scale factors for cropping (1.0=full, <1.0=zoom in, >1.0=zoom out)
            use_flip: Use horizontal flip TTA
            preserve_aspect_ratio: Pad to preserve AR before final resize
            padding_mode: How to pad
            min_resolution: Min dimension to trigger upscaling
            upscale_factor: Upscale ratio for low-res images
            use_square_crop: Add square center crop (good for aerial)
            fusion_method: How to fuse features
        """
        self.base_size = base_size  # FIXED - always output this size
        self.scales = scales
        self.use_flip = use_flip
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.padding_mode = padding_mode
        self.padding_value = padding_value
        self.min_resolution = min_resolution
        self.upscale_factor = upscale_factor
        self.use_square_crop = use_square_crop
        self.fusion_method = fusion_method
        
        self._print_config()
    
    def _print_config(self):
        print("\n" + "="*60)
        print("Advanced Multiscale Extractor (ViT Compatible)")
        print("="*60)
        print(f"  Fixed output size: {self.base_size} (ALWAYS)")
        print(f"  Crop scales: {self.scales}")
        print(f"  Use flip: {self.use_flip}")
        print(f"  Preserve aspect ratio: {self.preserve_aspect_ratio}")
        print(f"  Padding mode: {self.padding_mode}")
        print(f"  Min resolution threshold: {self.min_resolution}")
        print(f"  Use square crop: {self.use_square_crop}")
        print(f"  Fusion method: {self.fusion_method}")
        print("="*60 + "\n")
    
    def _enhance_low_resolution(self, images):
        """
        Enhance low resolution images by upscaling with bicubic interpolation.
        This happens BEFORE any other processing.
        """
        if images.dim() == 5:
            B, T, C, H, W = images.shape
        else:
            B, C, H, W = images.shape
        
        min_dim = min(H, W)
        
        if min_dim < self.min_resolution:
            scale = max(self.upscale_factor, self.min_resolution / min_dim)
            new_h = int(H * scale)
            new_w = int(W * scale)
            
            if images.dim() == 5:
                images_flat = images.view(B * T, C, H, W)
                enhanced = F.interpolate(
                    images_flat, 
                    size=(new_h, new_w), 
                    mode='bicubic', 
                    align_corners=False
                )
                return enhanced.view(B, T, C, new_h, new_w)
            else:
                return F.interpolate(
                    images, 
                    size=(new_h, new_w), 
                    mode='bicubic', 
                    align_corners=False
                )
        
        return images
    
    def _scale_crop(self, images, scale):
        """
        Apply scale-based cropping/padding.
        
        - scale < 1.0: Zoom IN (crop center region)
        - scale = 1.0: Full image
        - scale > 1.0: Zoom OUT (pad around image)
        
        Output is then resized to base_size.
        """
        if images.dim() == 5:
            B, T, C, H, W = images.shape
            is_video = True
            images_flat = images.view(B * T, C, H, W)
        else:
            B, C, H, W = images.shape
            is_video = False
            images_flat = images
        
        if scale == 1.0:
            # No cropping needed
            result = images_flat
        elif scale < 1.0:
            # Zoom IN: crop center region
            crop_h = int(H * scale)
            crop_w = int(W * scale)
            
            top = (H - crop_h) // 2
            left = (W - crop_w) // 2
            
            result = images_flat[:, :, top:top+crop_h, left:left+crop_w]
        else:
            # Zoom OUT: pad around image
            pad_h = int(H * (scale - 1.0) / 2)
            pad_w = int(W * (scale - 1.0) / 2)
            
            if self.padding_mode == 'constant':
                result = F.pad(images_flat, (pad_w, pad_w, pad_h, pad_h), 
                              mode='constant', value=self.padding_value)
            else:
                result = F.pad(images_flat, (pad_w, pad_w, pad_h, pad_h), 
                              mode=self.padding_mode)
        
        # Resize to FIXED base_size
        result = F.interpolate(result, size=self.base_size, mode='bilinear', align_corners=False)
        
        if is_video:
            return result.view(B, T, C, self.base_size[0], self.base_size[1])
        return result
    
    def _resize_preserve_aspect_ratio(self, images):
        """
        Resize while preserving aspect ratio using padding.
        Final output is ALWAYS base_size.
        """
        target_h, target_w = self.base_size
        
        if images.dim() == 5:
            B, T, C, H, W = images.shape
            is_video = True
            images_flat = images.view(B * T, C, H, W)
        else:
            B, C, H, W = images.shape
            is_video = False
            images_flat = images
        
        # Calculate scale to fit within target while preserving aspect ratio
        scale_h = target_h / H
        scale_w = target_w / W
        scale = min(scale_h, scale_w)
        
        new_h = int(H * scale)
        new_w = int(W * scale)
        
        # Resize preserving aspect ratio
        resized = F.interpolate(images_flat, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        # Calculate padding
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # Pad to target size
        if self.padding_mode == 'constant':
            padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom),
                          mode='constant', value=self.padding_value)
        else:
            # For reflect/replicate, handle edge cases
            if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom),
                              mode=self.padding_mode)
            else:
                padded = resized
        
        if is_video:
            return padded.view(B, T, C, target_h, target_w)
        return padded
    
    def _resize_standard(self, images):
        """Standard resize to base_size (may distort aspect ratio)."""
        if images.dim() == 5:
            B, T, C, H, W = images.shape
            images_flat = images.view(B * T, C, H, W)
            resized = F.interpolate(images_flat, size=self.base_size, mode='bilinear', align_corners=False)
            return resized.view(B, T, C, self.base_size[0], self.base_size[1])
        else:
            return F.interpolate(images, size=self.base_size, mode='bilinear', align_corners=False)
    
    def _extract_square_crop(self, images):
        """
        Extract square center crop, then resize to base_size.
        Good for aerial images which are often more square-ish.
        """
        if images.dim() == 5:
            B, T, C, H, W = images.shape
            is_video = True
            images_flat = images.view(B * T, C, H, W)
        else:
            B, C, H, W = images.shape
            is_video = False
            images_flat = images
        
        # Square crop (use smaller dimension)
        size = min(H, W)
        top = (H - size) // 2
        left = (W - size) // 2
        
        cropped = images_flat[:, :, top:top+size, left:left+size]
        
        # Resize to base_size (will stretch square to 2:1)
        # This actually helps model see "normalized" person shape
        result = F.interpolate(cropped, size=self.base_size, mode='bilinear', align_corners=False)
        
        if is_video:
            return result.view(B, T, C, self.base_size[0], self.base_size[1])
        return result
    
    def _horizontal_flip(self, images):
        """Apply horizontal flip."""
        return images.flip(dims=[-1])
    
    def _extract_features_single(self, model, images, use_gpu=True):
        """Extract features from processed images (already at base_size)."""
        if use_gpu:
            images = images.cuda()
        
        with torch.no_grad():
            feat = model(images)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            feat = F.normalize(feat, dim=1, p=2)
        
        return feat
    
    def extract_features(self, model, images, use_gpu=True):
        """
        Extract multi-scale features.
        ALL outputs are at FIXED base_size to be compatible with ViT.
        
        Args:
            model: ReID model
            images: (B, T, C, H, W) or (B, C, H, W) tensor
            use_gpu: Whether to use GPU
            
        Returns:
            Fused features (B, D)
        """
        all_features = []
        feature_weights = []
        
        # Step 1: Enhance low resolution images first
        enhanced_images = self._enhance_low_resolution(images)
        
        # Step 2: Multi-scale via cropping (all output base_size)
        for scale in self.scales:
            # Apply scale crop
            scaled = self._scale_crop(enhanced_images, scale)
            
            # Extract features - original
            feat = self._extract_features_single(model, scaled, use_gpu)
            all_features.append(feat)
            feature_weights.append(2.0 if scale == 1.0 else 1.0)
            
            # Horizontal flip
            if self.use_flip:
                flipped = self._horizontal_flip(scaled)
                feat_flip = self._extract_features_single(model, flipped, use_gpu)
                all_features.append(feat_flip)
                feature_weights.append(1.0 if scale == 1.0 else 0.8)
        
        # Step 3: Aspect ratio preserving version (output base_size)
        if self.preserve_aspect_ratio:
            ar_preserved = self._resize_preserve_aspect_ratio(enhanced_images)
            feat_ar = self._extract_features_single(model, ar_preserved, use_gpu)
            all_features.append(feat_ar)
            feature_weights.append(1.5)  # Important for aerial
            
            if self.use_flip:
                feat_ar_flip = self._extract_features_single(model, self._horizontal_flip(ar_preserved), use_gpu)
                all_features.append(feat_ar_flip)
                feature_weights.append(1.0)
        
        # Step 4: Square crop version (good for aerial, output base_size)
        if self.use_square_crop:
            square = self._extract_square_crop(enhanced_images)
            feat_sq = self._extract_features_single(model, square, use_gpu)
            all_features.append(feat_sq)
            feature_weights.append(1.5)  # Important for aerial
            
            if self.use_flip:
                feat_sq_flip = self._extract_features_single(model, self._horizontal_flip(square), use_gpu)
                all_features.append(feat_sq_flip)
                feature_weights.append(1.0)
        
        # Step 5: Fuse all features
        fused = self._fuse_features(all_features, feature_weights)
        
        return fused
    
    def _fuse_features(self, feature_list, weights=None):
        """
        Fuse features from multiple augmentations.
        """
        if len(feature_list) == 1:
            return feature_list[0]
        
        if self.fusion_method == 'mean':
            stacked = torch.stack(feature_list, dim=0)
            fused = stacked.mean(dim=0)
            
        elif self.fusion_method == 'weighted':
            if weights is None:
                weights = [1.0] * len(feature_list)
            
            weights = torch.tensor(weights, device=feature_list[0].device)
            weights = weights / weights.sum()
            
            stacked = torch.stack(feature_list, dim=0)
            fused = (stacked * weights.view(-1, 1, 1)).sum(dim=0)
            
        elif self.fusion_method == 'concat':
            fused = torch.cat(feature_list, dim=1)
            
        elif self.fusion_method == 'concat_norm':
            fused = torch.cat(feature_list, dim=1)
            fused = F.normalize(fused, dim=1, p=2)
            
        elif self.fusion_method == 'max':
            stacked = torch.stack(feature_list, dim=0)
            fused = stacked.max(dim=0)[0]
            
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        # Final normalization
        fused = F.normalize(fused, dim=1, p=2)
        
        return fused


def extract_features_advanced_multiscale(model, data_loader, extractor, use_gpu=True):
    """
    Extract features from dataloader using advanced multiscale extraction.
    """
    model.eval()
    
    all_features = []
    all_pids = []
    all_camids = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting features"):
            if len(batch) == 4:
                imgs, pids, camids, _ = batch
            elif len(batch) == 3:
                imgs, pids, camids = batch
            else:
                imgs, pids = batch[:2]
                camids = torch.zeros_like(pids)
            
            features = extractor.extract_features(model, imgs, use_gpu=use_gpu)
            
            all_features.append(features.cpu())
            all_pids.append(pids)
            all_camids.append(camids)
    
    all_features = torch.cat(all_features, dim=0)
    all_pids = torch.cat(all_pids, dim=0).numpy()
    all_camids = torch.cat(all_camids, dim=0).numpy()
    
    return all_features, all_pids, all_camids


# =============================================================================
# UFFM: Uncertainty Feature Fusion Method
# =============================================================================
class UFFM:
    """Uncertainty Feature Fusion Method"""
    
    def __init__(self, k=5, fusion_method='weighted_mean', temperature=0.1):
        self.k = k
        self.fusion_method = fusion_method
        self.temperature = temperature
    
    def compute_gallery_knn(self, gallery_features):
        gf_norm = F.normalize(gallery_features, dim=1, p=2)
        similarity = torch.matmul(gf_norm, gf_norm.t())
        similarity.fill_diagonal_(-float('inf'))
        knn_sim, knn_indices = torch.topk(similarity, k=self.k, dim=1)
        knn_distances = 1 - knn_sim
        return knn_indices, knn_distances, knn_sim
    
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
        
        urf = F.normalize(urf, dim=1, p=2)
        return urf
    
    def __call__(self, gallery_features):
        knn_indices, knn_distances, knn_similarities = self.compute_gallery_knn(gallery_features)
        urf = self.fuse_features(gallery_features, knn_indices, knn_similarities)
        return urf


# =============================================================================
# CCE & AMC
# =============================================================================
def compute_cce(q_camids, g_camids):
    q_camids = torch.tensor(q_camids).view(-1, 1)
    g_camids = torch.tensor(g_camids).view(1, -1)
    cce_matrix = (q_camids == g_camids).float()
    return cce_matrix


class AMC:
    """Auto-weighted Measure Combination"""
    
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
        distance_matrix = -S_combined
        
        return S_combined, distance_matrix


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
        tracklet_name = os.path.basename(tracklet_dir)
        return tracklet_name
    return "unknown"


def evaluate_metrics(distmat, q_pids, g_pids, q_camids, g_camids):
    num_q, num_g = distmat.shape
    index = torch.argsort(distmat, dim=1)
    index = index.numpy()
    
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
                  # UFFM params
                  uffm_k=5, uffm_fusion='weighted_mean', uffm_temperature=0.1,
                  # AMC params  
                  alpha=0.4, beta=0.4, gamma=0.2,
                  # Multiscale params
                  ms_config=None,
                  batch_size=4):
    """
    Evaluate model on a specific case.
    """
    print_section(f"Evaluating {case_name}")
    print(f"Subset: {case_subset}")
    
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
    
    # Use same transform as training - let multiscale extractor handle the rest
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
    
    query_loader = DataLoader(
        query_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False
    )
    gallery_loader = DataLoader(
        gallery_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False
    )
    
    print(f"Query set size: {len(query_dataset)}")
    print(f"Gallery set size: {len(gallery_dataset)}")
    
    # Sort tracklets
    query_info = [(i, t[1], extract_tracklet_name(t[0])) for i, t in enumerate(dataset.query)]
    query_info_sorted = sorted(query_info, key=lambda x: (x[1], x[2]))
    query_sorted_indices = [x[0] for x in query_info_sorted]
    query_tracklet_names = [x[2] for x in query_info_sorted]
    
    gallery_info = [(i, t[1], extract_tracklet_name(t[0])) for i, t in enumerate(dataset.gallery)]
    gallery_info_sorted = sorted(gallery_info, key=lambda x: (x[1], x[2]))
    gallery_sorted_indices = [x[0] for x in gallery_info_sorted]
    gallery_tracklet_names = [x[2] for x in gallery_info_sorted]
    
    # Initialize Multiscale Extractor
    if ms_config is None:
        ms_config = {}
    
    extractor = AdvancedMultiscaleExtractor(
        base_size=(cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1]),
        **ms_config
    )
    
    # Extract features
    model.eval()
    print("\nExtracting features...")
    start_time = time.time()
    
    with torch.no_grad():
        qf, q_pids, q_camids = extract_features_advanced_multiscale(
            model, query_loader, extractor, use_gpu=use_gpu
        )
        print(f"Query features: {qf.shape}")
        
        gf, g_pids, g_camids = extract_features_advanced_multiscale(
            model, gallery_loader, extractor, use_gpu=use_gpu
        )
        print(f"Gallery features: {gf.shape}")
    
    # Reorder
    qf_sorted = qf[query_sorted_indices]
    gf_sorted = gf[gallery_sorted_indices]
    q_pids_sorted = np.array([dataset.query[i][1] for i in query_sorted_indices])
    g_pids_sorted = np.array([dataset.gallery[i][1] for i in gallery_sorted_indices])
    q_camids_sorted = np.array([dataset.query[i][2] for i in query_sorted_indices])
    g_camids_sorted = np.array([dataset.gallery[i][2] for i in gallery_sorted_indices])
    
    # Apply UFFM
    print("\nApplying UFFM...")
    uffm = UFFM(k=uffm_k, fusion_method=uffm_fusion, temperature=uffm_temperature)
    gallery_urf = uffm(gf_sorted)
    
    # Apply AMC
    print("Computing AMC similarity...")
    amc = AMC(alpha=alpha, beta=beta, gamma=gamma)
    similarity_matrix, distance_matrix = amc(
        qf_sorted, gf_sorted, gallery_urf, q_camids_sorted, g_camids_sorted
    )
    
    # Ranking
    ranking_indices = torch.argsort(distance_matrix, dim=1).numpy()
    full_rankings = []
    for i in range(len(query_tracklet_names)):
        all_indices = ranking_indices[i]
        all_gallery_names = [gallery_tracklet_names[idx] for idx in all_indices]
        full_rankings.append(all_gallery_names)
    
    # Metrics
    try:
        cmc, mAP = evaluate_metrics(
            distance_matrix, q_pids_sorted, g_pids_sorted,
            q_camids_sorted, g_camids_sorted
        )
    except Exception as e:
        print(f"Warning: Metric evaluation failed ({e})")
        cmc = np.zeros(len(gallery_dataset))
        mAP = 0.0
    
    elapsed_time = time.time() - start_time
    ranks = [1, 5, 10, 20]
    
    print(f"\nEvaluation completed in {elapsed_time:.2f} seconds")
    print(f"\n{'Metric':<20} {'Value':<15}")
    print("-" * 35)
    print(f"{'mAP':<20} {mAP:.2%}")
    for r in ranks:
        if r <= len(cmc):
            print(f"{'Rank-' + str(r):<20} {cmc[r-1]:.2%}")
    
    return {
        'case_name': case_name,
        'case_subset': case_subset,
        'mAP': mAP,
        'cmc': cmc,
        'ranks': ranks,
        'num_query': len(query_dataset),
        'num_gallery': len(gallery_dataset),
        'elapsed_time': elapsed_time,
        'query_tracklet_names': query_tracklet_names,
        'full_rankings': full_rankings
    }


def print_summary(results):
    print_header("EVALUATION SUMMARY")
    
    print(f"\n{'Case':<30} {'mAP':<12} {'R-1':<12} {'R-5':<12} {'R-10':<12}")
    print("=" * 78)
    
    for result in results:
        case_name = result['case_name']
        mAP = result['mAP']
        cmc = result['cmc']
        
        r1 = cmc[0] if len(cmc) > 0 else 0
        r5 = cmc[4] if len(cmc) > 4 else cmc[-1] if len(cmc) > 0 else 0
        r10 = cmc[9] if len(cmc) > 9 else cmc[-1] if len(cmc) > 0 else 0
        
        print(f"{case_name:<30} {mAP:>10.2%} {r1:>10.2%} {r5:>10.2%} {r10:>10.2%}")
    
    num_cases = len(results)
    avg_map = sum([r['mAP'] for r in results]) / num_cases
    avg_r1 = sum([r['cmc'][0] for r in results]) / num_cases
    
    print("-" * 78)
    print(f"{'Average':<30} {avg_map:>10.2%} {avg_r1:>10.2%}")
    print_header(f"FINAL SCORE (Average Rank-1): {avg_r1:.2%}")


def generate_ranking_csv(results, output_path='evaluation_rankings.csv'):
    print_section("Generating Ranking CSV")
    
    csv_rows = []
    row_id = 1
    
    for result in results:
        case_name = result['case_name']
        query_names = result['query_tracklet_names']
        full_rankings = result['full_rankings']
        
        for query_name, full_galleries in zip(query_names, full_rankings):
            gallery_str = ' '.join(full_galleries)
            csv_rows.append({
                'row_id': row_id,
                'case': case_name,
                'query_tracklet': query_name,
                'ranked_gallery_tracklets': gallery_str
            })
            row_id += 1
    
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['row_id', 'case', 'query_tracklet', 'ranked_gallery_tracklets']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    
    print(f"CSV saved to: {output_path}")
    print(f"Total rows: {len(csv_rows)}")


# =============================================================================
# Preset Configurations for Aerial-Ground ReID
# =============================================================================
PRESETS = {
    'none': {
        'description': 'No multiscale - baseline',
        'config': {
            'scales': [1.0],
            'use_flip': False,
            'preserve_aspect_ratio': False,
            'use_square_crop': False,
            'fusion_method': 'mean'
        }
    },
    'flip_only': {
        'description': 'Only horizontal flip TTA',
        'config': {
            'scales': [1.0],
            'use_flip': True,
            'preserve_aspect_ratio': False,
            'use_square_crop': False,
            'fusion_method': 'mean'
        }
    },
    'fast': {
        'description': 'Fast - flip + aspect ratio preservation',
        'config': {
            'scales': [1.0],
            'use_flip': True,
            'preserve_aspect_ratio': True,
            'padding_mode': 'constant',
            'use_square_crop': False,
            'fusion_method': 'weighted'
        }
    },
    'balanced': {
        'description': 'Balanced - flip + AR + square crop (RECOMMENDED)',
        'config': {
            'scales': [1.0],
            'use_flip': True,
            'preserve_aspect_ratio': True,
            'padding_mode': 'reflect',
            'use_square_crop': True,
            'fusion_method': 'weighted',
            'min_resolution': 64,
            'upscale_factor': 1.5
        }
    },
    'aerial_optimized': {
        'description': 'Optimized for aerial with multi-scale crops',
        'config': {
            'scales': [0.9, 1.0, 1.1],
            'use_flip': True,
            'preserve_aspect_ratio': True,
            'padding_mode': 'reflect',
            'use_square_crop': True,
            'fusion_method': 'weighted',
            'min_resolution': 48,
            'upscale_factor': 2.0
        }
    },
    'best': {
        'description': 'Best accuracy - all augmentations',
        'config': {
            'scales': [0.85, 0.95, 1.0, 1.05, 1.15],
            'use_flip': True,
            'preserve_aspect_ratio': True,
            'padding_mode': 'reflect',
            'use_square_crop': True,
            'fusion_method': 'weighted',
            'min_resolution': 48,
            'upscale_factor': 2.0
        }
    }
}


def main():
    parser = argparse.ArgumentParser(description="AG-VPReID Evaluation - Advanced Multiscale (ViT Compatible)")
    parser.add_argument("--config_file", default="configs/adapter/vit_adapter.yml", type=str)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--cases", default="1,2,3", type=str)
    parser.add_argument("--output_file", default="", type=str)
    
    # Preset selection
    parser.add_argument("--preset", default="balanced", 
                        choices=list(PRESETS.keys()),
                        help="Preset configuration")
    
    # UFFM params
    parser.add_argument("--uffm_k", default=12, type=int)
    parser.add_argument("--uffm_fusion", default="weighted_mean", type=str)
    parser.add_argument("--uffm_temperature", default=0.2, type=float)
    
    # AMC params
    parser.add_argument("--alpha", default=0.4, type=float)
    parser.add_argument("--beta", default=0.5, type=float)
    parser.add_argument("--gamma", default=0.1, type=float)
    
    # Override multiscale params
    parser.add_argument("--ms_scales", default=None, type=str)
    parser.add_argument("--ms_no_flip", action="store_true")
    parser.add_argument("--ms_no_ar", action="store_true", help="Disable aspect ratio preservation")
    parser.add_argument("--ms_no_square", action="store_true", help="Disable square crop")
    parser.add_argument("--ms_fusion", default=None, type=str)
    
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    
    # Load preset config
    preset_name = args.preset
    ms_config = PRESETS[preset_name]['config'].copy()
    print(f"\nUsing preset: {preset_name}")
    print(f"Description: {PRESETS[preset_name]['description']}")
    
    # Apply overrides
    if args.ms_scales is not None:
        ms_config['scales'] = [float(s.strip()) for s in args.ms_scales.split(',')]
    if args.ms_no_flip:
        ms_config['use_flip'] = False
    if args.ms_no_ar:
        ms_config['preserve_aspect_ratio'] = False
    if args.ms_no_square:
        ms_config['use_square_crop'] = False
    if args.ms_fusion is not None:
        ms_config['fusion_method'] = args.ms_fusion
    
    # Load config
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    # CUDA
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        cudnn.benchmark = True
        torch.cuda.empty_cache()
    
    print_header("AG-VPReID Evaluation - Advanced Multiscale (ViT Compatible)")
    print(f"Model: {args.model_path}")
    print(f"Fixed output size: {cfg.INPUT.SIZE_TEST}")
    
    # Load model
    print_section("Loading Model")
    
    from datasets import data_manager
    dataset = data_manager.init_dataset(
        name=cfg.DATASETS.NAMES,
        root=cfg.DATASETS.ROOT_DIR,
        subset='case1_aerial_to_ground'
    )
    num_classes = dataset.num_train_pids
    camera_num = dataset.num_camera
    
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=0)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return
    
    checkpoint = torch.load(args.model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        sd = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        sd = checkpoint["model"]
    else:
        sd = checkpoint
    
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
          for k, v in sd.items()}
    
    model.load_state_dict(sd, strict=True)
    model.eval()
    
    if use_gpu:
        model = model.cuda()
    
    print("Model loaded!")
    
    # Evaluate
    cases_to_eval = [int(c.strip()) for c in args.cases.split(',')]
    all_cases = {
        1: {'name': 'Case 1: Aerial to Ground', 'subset': 'case1_aerial_to_ground'},
        2: {'name': 'Case 2: Ground to Aerial', 'subset': 'case2_ground_to_aerial'},
        3: {'name': 'Case 3: Aerial to Aerial', 'subset': 'case3_aerial_to_aerial'}
    }
    
    results = []
    total_start = time.time()
    
    for case_num in cases_to_eval:
        if case_num not in all_cases:
            continue
        
        case_info = all_cases[case_num]
        
        try:
            result = evaluate_case(
                cfg, model,
                case_info['name'],
                case_info['subset'],
                use_gpu=use_gpu,
                uffm_k=args.uffm_k,
                uffm_fusion=args.uffm_fusion,
                uffm_temperature=args.uffm_temperature,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                ms_config=ms_config,
                batch_size=args.batch_size
            )
            results.append(result)
        except Exception as e:
            print(f"\nERROR in {case_info['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    total_elapsed = time.time() - total_start
    
    if results:
        print_summary(results)
        print(f"\nTotal time: {timedelta(seconds=int(total_elapsed))}")
        
        csv_path = f'output/rankings_{preset_name}.csv'
        os.makedirs('output', exist_ok=True)
        generate_ranking_csv(results, output_path=csv_path)
        
        if args.output_file:
            import json
            output_data = {
                'model_path': args.model_path,
                'preset': preset_name,
                'ms_config': ms_config,
                'results': [{
                    'case': r['case_name'],
                    'mAP': float(r['mAP']),
                    'rank1': float(r['cmc'][0])
                } for r in results]
            }
            with open(args.output_file, 'w') as f:
                json.dump(output_data, f, indent=2)


if __name__ == '__main__':
    main()