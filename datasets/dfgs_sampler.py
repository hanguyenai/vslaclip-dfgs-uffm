"""
Depth-First Graph Sampler (DFGS) for Person Re-Identification.

Implementation based on the paper:
"CLIP-DFGS: A Hard Sample Mining Method for CLIP in Generalizable Person Re-Identification"
https://arxiv.org/pdf/2410.11255v1

The DFGS method uses depth-first search on a graph constructed from pairwise distances
to mine hard samples for metric learning.
"""

from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np
import torch


class DFGSSampler(Sampler):
    """
    Depth-First Graph Sampler (DFGS) for hard sample mining.
    
    This sampler constructs a graph based on pairwise distances between class features,
    then uses depth-first search to form mini-batches with hard samples.
    
    Args:
        data_source (list): List of tracklets (img_paths, pid, camid, ...)
        order_ids (list): List of person IDs in DFS order (based on feature similarity graph)
        id_to_indices (dict): Mapping from person ID to list of sample indices
        P (int): Number of IDs per batch
        K (int): Number of instances per ID
        k_neighbors (int): Number of nearest neighbors for graph construction (default: 10)
        m_difficulty (int): Difficulty coefficient - skip m most similar samples (default: 2)
        shuffle_graph (bool): Whether to shuffle graph nodes for diversity (default: True)
    """

    def __init__(self, data_source, order_ids, id_to_indices, P, K, 
                 k_neighbors=10, m_difficulty=2, shuffle_graph=True):
        self.data_source = data_source
        self.order_ids = order_ids
        self.id_to_indices = id_to_indices
        self.P = P  # Number of IDs per batch
        self.K = K  # Number of instances per ID
        self.batch_size = P * K
        self.k_neighbors = k_neighbors
        self.m_difficulty = m_difficulty
        self.shuffle_graph = shuffle_graph
        
        # Build index_dic for camera-aware sampling
        self.index_dic = defaultdict(list)
        self.camid_dic = defaultdict(lambda: defaultdict(list))  # pid -> camid -> indices
        
        for index, tracklet in enumerate(data_source):
            # Handle different tracklet formats
            if len(tracklet) >= 8:
                _, pid, camid, _, _, _, _, _ = tracklet
            elif len(tracklet) >= 4:
                _, pid, camid, _ = tracklet
            else:
                _, pid, camid = tracklet
            self.index_dic[pid].append(index)
            self.camid_dic[pid][camid].append(index)
        
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)
        
        # Estimate length
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.K:
                num = self.K
            self.length += num - num % self.K

    def _diff_cam_sample(self, pid, n):
        """
        Sample n instances from pid, preferring different cameras.
        This creates intra-class hard samples by selecting from different viewpoints.
        """
        all_indices = copy.deepcopy(self.index_dic[pid])
        
        if len(all_indices) <= n:
            # Not enough samples, use replacement
            return list(np.random.choice(all_indices, size=n, replace=True))
        
        # Try to sample from different cameras
        cam_indices = self.camid_dic[pid]
        available_cams = list(cam_indices.keys())
        
        if len(available_cams) == 1:
            # Only one camera, random sample
            return list(np.random.choice(all_indices, size=n, replace=False))
        
        selected = []
        cam_queue = copy.deepcopy(available_cams)
        random.shuffle(cam_queue)
        
        while len(selected) < n:
            if not cam_queue:
                cam_queue = copy.deepcopy(available_cams)
                random.shuffle(cam_queue)
            
            cam = cam_queue.pop(0)
            cam_samples = [idx for idx in cam_indices[cam] if idx not in selected]
            
            if cam_samples:
                selected.append(random.choice(cam_samples))
        
        return selected[:n]

    def __iter__(self):
        """
        Generate indices using depth-first search on the similarity graph.
        """
        final_idxs = []
        batch_idxs = []
        visited = set()
        
        # Copy order_ids and optionally shuffle for diversity
        available_pids = copy.deepcopy(self.order_ids)
        if self.shuffle_graph:
            random.shuffle(available_pids)
        
        # Initialize stack with random starting point
        if available_pids:
            start_pid = random.choice(available_pids)
            stack = [start_pid]
        else:
            stack = []
        
        pids_in_batch = set()
        
        while stack or available_pids:
            # If stack is empty but we still have unvisited pids, add a new starting point
            if not stack and available_pids:
                unvisited = [p for p in available_pids if p not in visited]
                if unvisited:
                    stack.append(random.choice(unvisited))
                else:
                    break
            
            if not stack:
                break
                
            # Depth-first: pop from stack
            pid = stack.pop()
            
            # Skip if already visited or not available
            if pid in visited:
                continue
            if pid not in self.index_dic:
                continue
                
            visited.add(pid)
            
            # Skip if this pid is already in the current batch
            if pid in pids_in_batch:
                continue
            
            # Sample K instances from this pid (preferring different cameras)
            sampled_indices = self._diff_cam_sample(pid, self.K)
            batch_idxs.extend(sampled_indices)
            pids_in_batch.add(pid)
            
            # Check if batch is complete
            if len(pids_in_batch) == self.P:
                final_idxs.extend(batch_idxs)
                batch_idxs = []
                pids_in_batch = set()
            
            # Find neighbors to add to stack (in reverse order for DFS)
            # order_ids should contain neighboring IDs in similarity order
            pid_idx = self.order_ids.index(pid) if pid in self.order_ids else -1
            if pid_idx >= 0:
                # Get neighbors from the graph structure
                neighbors = self._get_neighbors(pid_idx)
                for neighbor_pid in reversed(neighbors):
                    if neighbor_pid not in visited and neighbor_pid in self.index_dic:
                        stack.append(neighbor_pid)
        
        # Handle remaining samples in incomplete batch
        if batch_idxs and len(pids_in_batch) > 0:
            # Try to fill the batch with remaining pids
            remaining_pids = [p for p in self.pids if p not in visited and p not in pids_in_batch]
            random.shuffle(remaining_pids)
            
            for pid in remaining_pids:
                if len(pids_in_batch) >= self.P:
                    break
                sampled_indices = self._diff_cam_sample(pid, self.K)
                batch_idxs.extend(sampled_indices)
                pids_in_batch.add(pid)
            
            if len(pids_in_batch) == self.P:
                final_idxs.extend(batch_idxs)
        
        return iter(final_idxs)

    def _get_neighbors(self, pid_idx):
        """
        Get neighboring pids for a given pid index.
        Returns k_neighbors pids starting from m_difficulty offset.
        """
        neighbors = []
        start = max(0, pid_idx - self.k_neighbors // 2)
        end = min(len(self.order_ids), pid_idx + self.k_neighbors // 2 + 1)
        
        for i in range(start, end):
            if i != pid_idx:
                neighbors.append(self.order_ids[i])
        
        return neighbors[:self.k_neighbors]

    def __len__(self):
        return self.length


class DFGSSamplerWithGraph(Sampler):
    """
    Enhanced DFGS Sampler with explicit graph construction from pairwise distances.
    
    This version accepts a precomputed pairwise distance matrix and constructs
    the neighbor graph explicitly, following the paper more closely.
    
    Args:
        data_source (list): List of tracklets
        pairwise_distances (torch.Tensor or np.ndarray): N x N pairwise distance matrix
        pid_list (list): List of pids corresponding to rows/cols of distance matrix
        P (int): Number of IDs per batch
        K (int): Number of instances per ID
        k_neighbors (int): Number of nearest neighbors (k in the paper)
        m_difficulty (int): Difficulty offset - skip m easiest hard samples (m in the paper)
        shuffle_graph (bool): Whether to shuffle graph for diversity
    """
    
    def __init__(self, data_source, pairwise_distances, pid_list, P, K,
                 k_neighbors=10, m_difficulty=2, shuffle_graph=True):
        self.data_source = data_source
        self.P = P
        self.K = K
        self.batch_size = P * K
        self.k_neighbors = k_neighbors
        self.m_difficulty = m_difficulty
        self.shuffle_graph = shuffle_graph
        self.pid_list = pid_list
        
        # Convert distances to numpy if tensor
        if isinstance(pairwise_distances, torch.Tensor):
            pairwise_distances = pairwise_distances.cpu().numpy()
        self.pairwise_distances = pairwise_distances
        
        # Build index mappings
        self.index_dic = defaultdict(list)
        self.camid_dic = defaultdict(lambda: defaultdict(list))
        
        for index, tracklet in enumerate(data_source):
            if len(tracklet) >= 8:
                _, pid, camid, _, _, _, _, _ = tracklet
            elif len(tracklet) >= 4:
                _, pid, camid, _ = tracklet
            else:
                _, pid, camid = tracklet
            self.index_dic[pid].append(index)
            self.camid_dic[pid][camid].append(index)
        
        self.pids = list(self.index_dic.keys())
        
        # Build the graph: for each pid, store k nearest neighbors
        self.graph = self._build_graph()
        
        # Estimate length
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.K:
                num = self.K
            self.length += num - num % self.K

    def _build_graph(self):
        """
        Build neighbor graph based on pairwise distances.
        G[p] = {x_i_p | i = m+1, m+2, ..., m+k} (Equation 12 from paper)
        """
        graph = {}
        pid_to_idx = {pid: idx for idx, pid in enumerate(self.pid_list)}
        
        for pid in self.pids:
            if pid not in pid_to_idx:
                graph[pid] = []
                continue
                
            idx = pid_to_idx[pid]
            distances = self.pairwise_distances[idx].copy()
            
            # Set self-distance to infinity to exclude self
            distances[idx] = float('inf')
            
            # Get sorted indices (ascending by distance = most similar first)
            sorted_indices = np.argsort(distances)
            
            # Skip m most similar (too hard/noisy), take next k
            # G[p] = neighbors from index m to m+k
            start_idx = self.m_difficulty
            end_idx = self.m_difficulty + self.k_neighbors
            neighbor_indices = sorted_indices[start_idx:end_idx]
            
            # Convert indices back to pids
            neighbors = [self.pid_list[i] for i in neighbor_indices if i < len(self.pid_list)]
            graph[pid] = neighbors
        
        return graph

    def _diff_cam_sample(self, pid, n):
        """Sample n instances from pid, preferring different cameras."""
        all_indices = copy.deepcopy(self.index_dic[pid])
        
        if len(all_indices) <= n:
            return list(np.random.choice(all_indices, size=n, replace=True))
        
        cam_indices = self.camid_dic[pid]
        available_cams = list(cam_indices.keys())
        
        if len(available_cams) == 1:
            return list(np.random.choice(all_indices, size=n, replace=False))
        
        selected = []
        cam_queue = copy.deepcopy(available_cams)
        random.shuffle(cam_queue)
        
        while len(selected) < n:
            if not cam_queue:
                cam_queue = copy.deepcopy(available_cams)
                random.shuffle(cam_queue)
            
            cam = cam_queue.pop(0)
            cam_samples = [idx for idx in cam_indices[cam] if idx not in selected]
            
            if cam_samples:
                selected.append(random.choice(cam_samples))
        
        return selected[:n]

    def __iter__(self):
        """
        Algorithm 1 from the paper: Depth-First Graph Sampler
        """
        final_idxs = []
        batch_idxs = []
        
        # Create availability tracking
        available_pids = set(self.pids)
        
        # Shuffle graph neighbors for diversity (line 4-5 in Algorithm 1)
        if self.shuffle_graph:
            for pid in self.graph:
                random.shuffle(self.graph[pid])
        
        # Initialize stack with random starting pid (line 3)
        stack = [random.choice(list(available_pids))] if available_pids else []
        
        pids_in_batch = set()
        
        while stack or available_pids:
            # If stack empty but pids remain, restart from random available pid
            if not stack:
                remaining = list(available_pids)
                if remaining:
                    stack.append(random.choice(remaining))
                else:
                    break
            
            # Pop from stack (line 7: depth-first search)
            pid = stack.pop()
            
            # Check availability (line 8-9)
            if pid not in available_pids:
                continue
            
            # Skip if already in current batch (ensures P unique pids per batch)
            if pid in pids_in_batch:
                continue
            
            # Sample K instances from different cameras (line 10)
            sampled_indices = self._diff_cam_sample(pid, self.K)
            batch_idxs.extend(sampled_indices)
            pids_in_batch.add(pid)
            available_pids.discard(pid)
            
            # Check if batch is complete (line 11-13)
            if len(pids_in_batch) == self.P:
                final_idxs.extend(batch_idxs)
                batch_idxs = []
                pids_in_batch = set()
            
            # Add neighbors to stack in reverse order (line 14-16)
            neighbors = self.graph.get(pid, [])
            for neighbor in reversed(neighbors):
                if neighbor in available_pids:
                    stack.append(neighbor)
        
        return iter(final_idxs)

    def __len__(self):
        return self.length

    def update_distances(self, new_distances):
        """
        Update pairwise distances and rebuild graph.
        Call this at the start of each epoch for DFGS_I(.) variant.
        """
        if isinstance(new_distances, torch.Tensor):
            new_distances = new_distances.cpu().numpy()
        self.pairwise_distances = new_distances
        self.graph = self._build_graph()


def build_id_to_indices(data_source):
    """
    Build a mapping from person ID to list of sample indices.
    
    Args:
        data_source: List of tracklets (img_paths, pid, camid, ...)
    
    Returns:
        dict: Mapping from pid to list of indices
    """
    id_to_indices = defaultdict(list)
    
    for index, tracklet in enumerate(data_source):
        if len(tracklet) >= 8:
            _, pid, _, _, _, _, _, _ = tracklet
        elif len(tracklet) >= 4:
            _, pid, _, _ = tracklet
        else:
            _, pid, _ = tracklet
        id_to_indices[pid].append(index)
    
    return dict(id_to_indices)


def compute_pairwise_distances(features, metric='euclidean'):
    """
    Compute pairwise distance matrix from features.
    
    Args:
        features (torch.Tensor): N x D feature matrix
        metric (str): Distance metric ('euclidean' or 'cosine')
    
    Returns:
        torch.Tensor: N x N distance matrix
    """
    if metric == 'euclidean':
        # Euclidean distance
        diff = features.unsqueeze(0) - features.unsqueeze(1)
        distances = torch.norm(diff, dim=2)
    elif metric == 'cosine':
        # Cosine distance = 1 - cosine_similarity
        features_norm = features / features.norm(dim=1, keepdim=True)
        similarity = torch.mm(features_norm, features_norm.t())
        distances = 1 - similarity
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return distances

class GS(Sampler):
    def __init__(self, data_source, batch_size, num_instances, model):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.model = model
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(dict)
        self.index_dic_imgpath = defaultdict(list)

        for index, data in enumerate(self.data_source):
            if data[2] in self.index_dic[data[1]].keys():
                self.index_dic[data[1]][data[2]].append(index)
            else:
                self.index_dic[data[1]][data[2]] = [index]
            self.index_dic_imgpath[data[1]].append(data[0])
            self.domain_info[data[-1].lower()] += 1
        self.pids = list(self.index_dic.keys())

        self.length = 0
        for pid in self.pids:
            num = sum([len(self.index_dic[pid][key]) for key in self.index_dic[pid].keys()])
            self.length += num

    def sort_dic(self, s):
        ks = list(s.keys())
        len_k = np.array([len(s[k]) for k in s.keys()])
        ix = len_k.argsort()[::-1]
        return {ks[i]: s[ks[i]] for i in ix}

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        pids = copy.deepcopy(self.pids)
        random.shuffle(pids)
        print("Dist Updating!")
        for pid in pids:
            dic_tmp = copy.deepcopy(self.index_dic[pid])

            cids = list(dic_tmp.keys())
            for cid in cids:
                random.shuffle(dic_tmp[cid])
            idxs = []
            while cids:
                num = 0
                dic_tmp = self.sort_dic(dic_tmp)
                for cid in cids:
                    num += 1
                    idxs.append(dic_tmp[cid].pop())
                    if len(dic_tmp[cid]) == 0:
                        cids.remove(cid)
                    if num == self.num_instances:
                        break
            if len(idxs) <= 1:
                continue
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)

            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        final_idxs = []
        transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

        model = copy.deepcopy(self.model).cuda().eval()
        index_dic = defaultdict(list)
        for index, data in enumerate(self.data_source):
            index_dic[data[1]].append(index)
        pids = list(index_dic.keys())
        inex_dic = {k: index_dic[k][random.randint(0, len(index_dic[k]) - 1)] for k in pids}
        feat_dist = {}
        choice_set = CommDataset([self.data_source[i] for i in list(inex_dic.values())], transforms, relabel=False)
        choice_loader = DataLoader(
            choice_set, batch_size=256, shuffle=False, num_workers=8,
            collate_fn=val_collate_fn
        )
        feats = torch.tensor([]).cuda()
        for i, (img, _, _) in enumerate(choice_loader):
            with torch.no_grad():
                img = img.cuda()
                feat = model(img)
                feats = torch.cat((feats, feat), dim=0)

        dist_mat = euclidean_dist(feats, feats)
        for i in range(len(dist_mat)):
            dist_mat[i][i] = float("inf")

        for i, feat in enumerate(dist_mat):
            loc = torch.argsort(feat)
            feat_dist[pids[i]] = [pids[int(loc[j].cpu())] for j in range(31)]

        random.shuffle(pids)
        i = 0
        for k in pids:
            if i > 24320:
                break
            i += 1
            v = feat_dist[k]
            final_idxs.extend(batch_idxs_dict[k][random.randint(0, len(batch_idxs_dict[k]) - 1)])
            for k in v:
                i += 1
                final_idxs.extend(batch_idxs_dict[k][random.randint(0, len(batch_idxs_dict[k]) - 1)])
        self.length = len(final_idxs)
        print(self.length)
        return iter(final_idxs)

    def __len__(self):
        return self.length