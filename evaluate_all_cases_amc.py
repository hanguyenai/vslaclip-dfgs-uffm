"""
Comprehensive evaluation script for AG-VPReID dataset with AMC and CFFM
Evaluates the model on all three cases:
- Case 1: Aerial to Ground
- Case 2: Ground to Aerial
- Case 3: Aerial to Aerial

Enhanced inference techniques from paper:
- CFFM (Certain Feature Fusion Method): KNN-based gallery feature refinement
- CCE (Camera Consistency Encoding): Camera-aware similarity
- AMC (Auto-weighted Measure Combination): Weighted combination of multiple similarities
  S* = α·S(q, gj) + β·S(q, URFj) + γ·CCE(q, gj)
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

# Add current directory to path
sys.path.append('.')

from config import cfg
from datasets.make_video_dataloader import make_dataloader
from model.make_model_clipvideoreid_reidadapter_pbp import make_model
from utils.test_video_reid import _feats_of_loader, extract_feat_sampled_frames


# =============================================================================
# CFFM: Certain Feature Fusion Method (Uncertainty Feature Fusion)
# =============================================================================
class CFFM:
    """
    Certain Feature Fusion Method (CFFM) / Uncertainty Feature Fusion
    
    For each gallery feature, find its K nearest neighbors in the gallery
    and fuse them to create Uncertainty-Refined Features (URF).
    
    URF_j = weighted_mean(gj, knn(gj))
    
    This helps to:
    1. Reduce noise in gallery features
    2. Leverage local structure in the feature space
    3. Create more robust gallery representations
    """
    
    def __init__(self, k=5, fusion_method='weighted_mean', temperature=0.1):
        """
        Args:
            k: Number of nearest neighbors to use
            fusion_method: How to fuse neighbors ('mean', 'weighted_mean', 'attention')
            temperature: Temperature for weighted fusion (lower = sharper weights)
        """
        self.k = k
        self.fusion_method = fusion_method
        self.temperature = temperature
    
    def compute_gallery_knn(self, gallery_features):
        """
        Compute KNN indices for each gallery feature within the gallery itself.
        
        Args:
            gallery_features: (N, D) tensor of gallery features
            
        Returns:
            knn_indices: (N, K) tensor of KNN indices for each gallery
            knn_distances: (N, K) tensor of distances to KNN
        """
        # Normalize features
        gf_norm = F.normalize(gallery_features, dim=1, p=2)
        
        # Compute pairwise similarity (cosine)
        similarity = torch.matmul(gf_norm, gf_norm.t())  # (N, N)
        
        # Exclude self (set diagonal to -inf)
        similarity.fill_diagonal_(-float('inf'))
        
        # Get top-K neighbors (highest similarity = nearest)
        knn_sim, knn_indices = torch.topk(similarity, k=self.k, dim=1)
        
        # Convert similarity to distance for weighting
        knn_distances = 1 - knn_sim  # cosine distance
        
        return knn_indices, knn_distances, knn_sim
    
    def fuse_features(self, gallery_features, knn_indices, knn_similarities):
        """
        Fuse gallery features with their KNN to create URF (Uncertainty-Refined Features).
        
        Args:
            gallery_features: (N, D) tensor
            knn_indices: (N, K) tensor
            knn_similarities: (N, K) tensor
            
        Returns:
            urf: (N, D) tensor of uncertainty-refined features
        """
        N, D = gallery_features.shape
        K = knn_indices.shape[1]
        
        if self.fusion_method == 'mean':
            # Simple mean of neighbors
            neighbor_features = gallery_features[knn_indices.view(-1)].view(N, K, D)
            urf = neighbor_features.mean(dim=1)
            
        elif self.fusion_method == 'weighted_mean':
            # Weighted mean based on similarity
            # Apply temperature scaling for sharper/softer weights
            weights = F.softmax(knn_similarities / self.temperature, dim=1)  # (N, K)
            
            # Gather neighbor features
            neighbor_features = gallery_features[knn_indices.view(-1)].view(N, K, D)
            
            # Weighted sum
            urf = (neighbor_features * weights.unsqueeze(-1)).sum(dim=1)
            
        elif self.fusion_method == 'self_and_neighbors':
            # Include original feature + weighted neighbors
            weights = F.softmax(knn_similarities / self.temperature, dim=1)
            neighbor_features = gallery_features[knn_indices.view(-1)].view(N, K, D)
            neighbor_contrib = (neighbor_features * weights.unsqueeze(-1)).sum(dim=1)
            
            # Combine: self + neighbors (with self having higher weight)
            self_weight = 0.5
            neighbor_weight = 0.5
            urf = self_weight * gallery_features + neighbor_weight * neighbor_contrib
            
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        # Normalize URF
        urf = F.normalize(urf, dim=1, p=2)
        
        return urf
    
    def __call__(self, gallery_features):
        """
        Apply CFFM to gallery features.
        
        Args:
            gallery_features: (N, D) tensor
            
        Returns:
            urf: (N, D) tensor of uncertainty-refined features
        """
        knn_indices, knn_distances, knn_similarities = self.compute_gallery_knn(gallery_features)
        urf = self.fuse_features(gallery_features, knn_indices, knn_similarities)
        return urf


# =============================================================================
# CCE: Camera Consistency Encoding
# =============================================================================
def compute_cce(q_camids, g_camids):
    """
    Compute Camera Consistency Encoding (CCE) matrix.
    
    CCE(q, gj) = 1 if Cam(q) == Cam(gj), else 0
    
    Args:
        q_camids: (num_query,) array of query camera IDs
        g_camids: (num_gallery,) array of gallery camera IDs
        
    Returns:
        cce_matrix: (num_query, num_gallery) tensor
            1 where cameras match, 0 otherwise
    """
    q_camids = torch.tensor(q_camids).view(-1, 1)  # (Q, 1)
    g_camids = torch.tensor(g_camids).view(1, -1)  # (1, G)
    
    cce_matrix = (q_camids == g_camids).float()  # (Q, G)
    
    return cce_matrix


# =============================================================================
# AMC: Auto-weighted Measure Combination
# =============================================================================
class AMC:
    """
    Auto-weighted Measure Combination (AMC)
    
    Combines multiple similarity measures:
    S* = α·S(q, gj) + β·S(q, URFj) + γ·CCE(q, gj)
    
    Where:
    - S(q, gj): Direct cosine similarity between query and gallery
    - S(q, URFj): Cosine similarity between query and uncertainty-refined gallery
    - CCE(q, gj): Camera consistency encoding (1 if same camera, 0 otherwise)
    """
    
    def __init__(self, alpha=0.4, beta=0.4, gamma=0.2, learn_weights=False):
        """
        Args:
            alpha: Weight for direct similarity S(q, gj)
            beta: Weight for URF similarity S(q, URFj)
            gamma: Weight for CCE
            learn_weights: If True, learn weights from data (not implemented)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.learn_weights = learn_weights
        
        # Normalize weights
        total = alpha + beta + gamma
        self.alpha /= total
        self.beta /= total
        self.gamma /= total
    
    def compute_similarity_matrix(self, query_features, gallery_features):
        """
        Compute cosine similarity matrix.
        
        Args:
            query_features: (Q, D) tensor
            gallery_features: (G, D) tensor
            
        Returns:
            similarity: (Q, G) tensor
        """
        qf_norm = F.normalize(query_features, dim=1, p=2)
        gf_norm = F.normalize(gallery_features, dim=1, p=2)
        
        similarity = torch.matmul(qf_norm, gf_norm.t())
        
        return similarity
    
    def __call__(self, query_features, gallery_features, gallery_urf, q_camids, g_camids):
        """
        Compute AMC similarity matrix.
        
        Args:
            query_features: (Q, D) tensor
            gallery_features: (G, D) tensor
            gallery_urf: (G, D) tensor of uncertainty-refined gallery features
            q_camids: (Q,) array of query camera IDs
            g_camids: (G,) array of gallery camera IDs
            
        Returns:
            final_similarity: (Q, G) tensor (higher = more similar)
            distance_matrix: (Q, G) tensor (lower = more similar, for ranking)
        """
        # S(q, gj): Direct similarity
        S_direct = self.compute_similarity_matrix(query_features, gallery_features)
        
        # S(q, URFj): URF similarity
        S_urf = self.compute_similarity_matrix(query_features, gallery_urf)
        
        # CCE(q, gj): Camera consistency encoding
        cce = compute_cce(q_camids, g_camids)
        
        # Combine with AMC formula
        # S* = α·S(q, gj) + β·S(q, URFj) + γ·CCE(q, gj)
        S_combined = (
            self.alpha * S_direct + 
            self.beta * S_urf + 
            self.gamma * cce
        )
        
        # Convert to distance (lower = better for ranking)
        distance_matrix = -S_combined
        
        return S_combined, distance_matrix


# =============================================================================
# Utility Functions
# =============================================================================
def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_section(text):
    """Print formatted section"""
    print("\n" + "-" * 80)
    print(f"  {text}")
    print("-" * 80)


def extract_tracklet_name(img_paths):
    """Extract tracklet name from image paths"""
    if isinstance(img_paths, (list, tuple)) and len(img_paths) > 0:
        tracklet_dir = os.path.dirname(img_paths[0])
        tracklet_name = os.path.basename(tracklet_dir)
        return tracklet_name
    return "unknown"


def evaluate_metrics(distmat, q_pids, g_pids, q_camids, g_camids):
    """Evaluate CMC and mAP"""
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
        
        # Compute AP and CMC
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
# Main Evaluation Function with AMC + CFFM
# =============================================================================
def evaluate_case_amc(cfg, model, case_name, case_subset, use_gpu=True,
                      cffm_k=5, cffm_fusion='weighted_mean', cffm_temperature=0.1,
                      alpha=0.4, beta=0.4, gamma=0.2, batch_size=4):
    """
    Evaluate model on a specific case with AMC and CFFM.
    
    Args:
        cfg: Configuration object
        model: The trained model
        case_name: Name of the case
        case_subset: Subset identifier
        use_gpu: Whether to use GPU
        cffm_k: Number of neighbors for CFFM
        cffm_fusion: Fusion method for CFFM
        cffm_temperature: Temperature for weighted fusion
        alpha: Weight for S(q, gj) in AMC
        beta: Weight for S(q, URFj) in AMC
        gamma: Weight for CCE in AMC
        batch_size: Batch size for feature extraction
    """
    print_section(f"Evaluating {case_name}")
    print(f"Subset: {case_subset}")
    print(f"CFFM: k={cffm_k}, fusion={cffm_fusion}, temp={cffm_temperature}")
    print(f"AMC weights: α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}")
    
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
    
    # Create transformations
    spatial_transform_test = ST.Compose([
        ST.Scale((cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1]), interpolation=3),
        ST.ToTensor(),
        ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    temporal_transform_test = TT.TemporalRestrictedBeginCrop(size=cfg.TEST.SEQ_LEN)
    
    # Create datasets and loaders
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
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"Query set size: {len(query_dataset)}")
    print(f"Gallery set size: {len(gallery_dataset)}")
    
    # Sort tracklets by PID and name
    print("Sorting tracklets...")
    
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
        qf, q_pids, q_camids = _feats_of_loader(
            model, query_loader, extract_feat_sampled_frames, use_gpu=use_gpu
        )
        print(f"Query features: {qf.shape}")
        
        gf, g_pids, g_camids = _feats_of_loader(
            model, gallery_loader, extract_feat_sampled_frames, use_gpu=use_gpu
        )
        print(f"Gallery features: {gf.shape}")
    
    # Reorder features according to sorted indices
    qf_sorted = qf[query_sorted_indices]
    gf_sorted = gf[gallery_sorted_indices]
    q_pids_sorted = np.array([dataset.query[i][1] for i in query_sorted_indices])
    g_pids_sorted = np.array([dataset.gallery[i][1] for i in gallery_sorted_indices])
    q_camids_sorted = np.array([dataset.query[i][2] for i in query_sorted_indices])
    g_camids_sorted = np.array([dataset.gallery[i][2] for i in gallery_sorted_indices])
    
    # Apply CFFM to gallery features
    print("\nApplying CFFM (Uncertainty Feature Fusion)...")
    cffm = CFFM(k=cffm_k, fusion_method=cffm_fusion, temperature=cffm_temperature)
    gallery_urf = cffm(gf_sorted)
    print(f"Gallery URF features: {gallery_urf.shape}")
    
    # Apply AMC for final similarity
    print("\nComputing AMC similarity...")
    amc = AMC(alpha=alpha, beta=beta, gamma=gamma)
    similarity_matrix, distance_matrix = amc(
        qf_sorted, gf_sorted, gallery_urf, q_camids_sorted, g_camids_sorted
    )
    
    # Get ranking indices
    ranking_indices = torch.argsort(distance_matrix, dim=1).numpy()
    
    # Build full ranking list
    full_rankings = []
    for i in range(len(query_tracklet_names)):
        all_indices = ranking_indices[i]
        all_gallery_names = [gallery_tracklet_names[idx] for idx in all_indices]
        full_rankings.append(all_gallery_names)
    
    # Compute metrics
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
    
    # Print results
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
        'num_query_ids': len(set([x[1] for x in dataset.query])),
        'num_gallery_ids': len(set([x[1] for x in dataset.gallery])),
        'elapsed_time': elapsed_time,
        'query_tracklet_names': query_tracklet_names,
        'full_rankings': full_rankings
    }


def print_summary(results):
    """Print summary of all results"""
    print_header("EVALUATION SUMMARY")
    
    print(f"\n{'Case':<30} {'mAP':<12} {'R-1':<12} {'R-5':<12} {'R-10':<12}")
    print("=" * 78)
    
    total_map = 0
    total_r1 = 0
    total_r5 = 0
    total_r10 = 0
    
    for result in results:
        case_name = result['case_name']
        mAP = result['mAP']
        cmc = result['cmc']
        
        r1 = cmc[0] if len(cmc) > 0 else 0
        r5 = cmc[4] if len(cmc) > 4 else cmc[-1] if len(cmc) > 0 else 0
        r10 = cmc[9] if len(cmc) > 9 else cmc[-1] if len(cmc) > 0 else 0
        
        print(f"{case_name:<30} {mAP:>10.2%} {r1:>10.2%} {r5:>10.2%} {r10:>10.2%}")
        
        total_map += mAP
        total_r1 += r1
        total_r5 += r5
        total_r10 += r10
    
    num_cases = len(results)
    print("-" * 78)
    print(f"{'Average':<30} {total_map/num_cases:>10.2%} {total_r1/num_cases:>10.2%} {total_r5/num_cases:>10.2%} {total_r10/num_cases:>10.2%}")
    
    avg_rank1 = sum([r['cmc'][0] for r in results]) / num_cases
    print_header(f"FINAL SCORE (Average Rank-1): {avg_rank1:.2%}")


def generate_ranking_csv(results, output_path='evaluation_rankings.csv'):
    """Generate CSV file with rankings"""
    print_section("Generating Ranking CSV")
    
    csv_rows = []
    row_id = 1
    
    for result in results:
        case_name = result['case_name']
        query_names = result['query_tracklet_names']
        full_rankings = result['full_rankings']
        
        print(f"Processing {case_name}: {len(query_names)} queries")
        
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
    
    print(f"CSV file saved to: {output_path}")
    print(f"Total rows: {len(csv_rows)}")


def main():
    parser = argparse.ArgumentParser(description="AG-VPReID Evaluation with AMC + CFFM")
    parser.add_argument(
        "--config_file",
        default="configs/adapter/vit_adapter.yml",
        help="path to config file",
        type=str
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="path to trained model checkpoint",
        type=str
    )
    parser.add_argument(
        "--cases",
        default="1,2,3",
        help="which cases to evaluate (e.g., '1,2,3' or '1,3')",
        type=str
    )
    parser.add_argument(
        "--output_file",
        default="",
        help="path to save evaluation results (optional)",
        type=str
    )
    # CFFM parameters
    parser.add_argument(
        "--cffm_k",
        default=12,
        help="Number of neighbors for CFFM (default: 5)",
        type=int
    )
    parser.add_argument(
        "--cffm_fusion",
        default="weighted_mean",
        choices=["mean", "weighted_mean", "self_and_neighbors"],
        help="Fusion method for CFFM",
        type=str
    )
    parser.add_argument(
        "--cffm_temperature",
        default=0.2,
        help="Temperature for CFFM weighted fusion (default: 0.1)",
        type=float
    )
    # AMC parameters
    parser.add_argument(
        "--alpha",
        default=0.4,
        help="Weight for S(q, gj) in AMC (default: 0.4)",
        type=float
    )
    parser.add_argument(
        "--beta",
        default=0.5,
        help="Weight for S(q, URFj) in AMC (default: 0.4)",
        type=float
    )
    parser.add_argument(
        "--gamma",
        default=0.1,
        help="Weight for CCE in AMC (default: 0.2)",
        type=float
    )
    parser.add_argument(
        "--batch_size",
        default=4,
        help="Batch size for feature extraction (default: 4)",
        type=int
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    # Check CUDA
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        cudnn.benchmark = True
        torch.cuda.empty_cache()
    else:
        print("CUDA not available, using CPU")
    
    print_header("AG-VPReID Evaluation with AMC + CFFM")
    print(f"Config file: {args.config_file}")
    print(f"Model path: {args.model_path}")
    print(f"Dataset root: {cfg.DATASETS.ROOT_DIR}")
    print(f"\nCFFM settings:")
    print(f"  K (neighbors): {args.cffm_k}")
    print(f"  Fusion method: {args.cffm_fusion}")
    print(f"  Temperature: {args.cffm_temperature}")
    print(f"\nAMC settings:")
    print(f"  α (direct similarity): {args.alpha}")
    print(f"  β (URF similarity): {args.beta}")
    print(f"  γ (CCE): {args.gamma}")
    print(f"\nBatch size: {args.batch_size}")
    
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
    view_num = 0
    
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found at {args.model_path}")
        return
    
    print(f"Loading checkpoint from: {args.model_path}")
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
    
    print("Model loaded successfully!")
    
    # Parse cases
    cases_to_eval = [int(c.strip()) for c in args.cases.split(',')]
    
    all_cases = {
        1: {'name': 'Case 1: Aerial to Ground', 'subset': 'case1_aerial_to_ground'},
        2: {'name': 'Case 2: Ground to Aerial', 'subset': 'case2_ground_to_aerial'},
        3: {'name': 'Case 3: Aerial to Aerial', 'subset': 'case3_aerial_to_aerial'}
    }
    
    # Evaluate
    results = []
    total_start_time = time.time()
    
    for case_num in cases_to_eval:
        if case_num not in all_cases:
            print(f"Warning: Invalid case number {case_num}, skipping...")
            continue
        
        case_info = all_cases[case_num]
        
        try:
            result = evaluate_case_amc(
                cfg, model,
                case_info['name'],
                case_info['subset'],
                use_gpu=use_gpu,
                cffm_k=args.cffm_k,
                cffm_fusion=args.cffm_fusion,
                cffm_temperature=args.cffm_temperature,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                batch_size=args.batch_size
            )
            results.append(result)
        except Exception as e:
            print(f"\nERROR in {case_info['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    total_elapsed = time.time() - total_start_time
    
    if results:
        print_summary(results)
        print(f"\nTotal evaluation time: {timedelta(seconds=int(total_elapsed))}")
        
        # Generate CSV
        csv_output_path = 'output/evaluation_rankings_amc_epoch60_bicubic.csv'
        os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
        generate_ranking_csv(results, output_path=csv_output_path)
        
        # Save results
        if args.output_file:
            import json
            output_data = {
                'model_path': args.model_path,
                'config_file': args.config_file,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'settings': {
                    'cffm_k': args.cffm_k,
                    'cffm_fusion': args.cffm_fusion,
                    'cffm_temperature': args.cffm_temperature,
                    'alpha': args.alpha,
                    'beta': args.beta,
                    'gamma': args.gamma
                },
                'results': [{
                    'case_name': r['case_name'],
                    'mAP': float(r['mAP']),
                    'rank1': float(r['cmc'][0]),
                    'rank5': float(r['cmc'][4]) if len(r['cmc']) > 4 else None,
                    'rank10': float(r['cmc'][9]) if len(r['cmc']) > 9 else None
                } for r in results]
            }
            with open(args.output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output_file}")
    else:
        print("\nNo results to display.")


if __name__ == '__main__':
    main()