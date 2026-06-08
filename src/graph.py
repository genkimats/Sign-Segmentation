import numpy as np

class SkeletonGraph:
    def __init__(self, num_vertices=65):
        self.num_vertices = num_vertices
        
        # Define indices for each anatomical part
        self.body_indices = list(range(0, 23))
        self.lh_indices = list(range(23, 44))
        self.rh_indices = list(range(44, 65))
        
        # 1. Decoupled Matrices (for DecoupledSTGCNBlock)
        self.A_body = self._get_subgraph_adjacency(self.body_indices, self._get_body_edges())
        self.A_lh = self._get_subgraph_adjacency(self.lh_indices, self._get_hand_edges(offset=23))
        self.A_rh = self._get_subgraph_adjacency(self.rh_indices, self._get_hand_edges(offset=44))
        
        # 2. Unified Matrix (for standard STGCNBlock)
        # This fixes the AttributeError
        self.A = self._get_subgraph_adjacency(list(range(num_vertices)), self._get_all_edges())

    def _get_body_edges(self):
        return [
            (0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8), 
            (9,10), (11,12), 
            (11,13), (13,15), (15,17), (15,19), (15,21), (17,19), 
            (12,14), (14,16), (16,18), (16,20), (16,22), (18,20)
        ]

    def _get_hand_edges(self, offset):
        hand_edges = [
            (0,1), (1,2), (2,3), (3,4),       # Thumb
            (0,5), (5,6), (6,7), (7,8),       # Index
            (0,9), (9,10), (10,11), (11,12),  # Middle
            (0,13), (13,14), (14,15), (15,16),# Ring
            (0,17), (17,18), (18,19), (19,20) # Pinky
        ]
        return [(i + offset, j + offset) for i, j in hand_edges]

    def _get_all_edges(self):
        """Combines all edges to build the full skeleton graph."""
        edges = self._get_body_edges()
        edges.extend(self._get_hand_edges(offset=23))
        edges.extend(self._get_hand_edges(offset=44))
        # Connect hands to arms
        edges.append((15, 23)) 
        edges.append((16, 44))
        return edges

    def _get_subgraph_adjacency(self, node_indices, edges):
        num_nodes = len(node_indices)
        A = np.zeros((num_nodes, num_nodes))
        
        idx_map = {global_idx: local_idx for local_idx, global_idx in enumerate(node_indices)}
        
        for i, j in edges:
            if i in idx_map and j in idx_map:
                local_i, local_j = idx_map[i], idx_map[j]
                A[local_i, local_j] = 1
                A[local_j, local_i] = 1
                
        A = A + np.eye(num_nodes)
        D = np.diag(np.sum(A, axis=1) ** -0.5)
        A_normalized = D @ A @ D
        return A_normalized