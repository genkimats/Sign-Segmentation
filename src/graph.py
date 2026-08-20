import numpy as np

class SkeletonGraph:
    # --- Must match extract_face_keypoints.py's SELECTED_INDICES construction
    # exactly -- these determine which array position each face landmark ends
    # up at, and therefore which local vertex id each face edge below refers to.
    _LIPS_INDICES = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308
    ]
    _LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    _RIGHT_EYE_INDICES = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
    _LEFT_EYEBROW_INDICES = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    _RIGHT_EYEBROW_INDICES = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
    _FACE_SELECTED_INDICES = sorted(list(set(
        _LIPS_INDICES + _LEFT_EYE_INDICES + _RIGHT_EYE_INDICES + _LEFT_EYEBROW_INDICES + _RIGHT_EYEBROW_INDICES
    )))
    NUM_FACE_VERTICES = len(_FACE_SELECTED_INDICES)  # 83 -- must match queue_train.py's NUM_FACE_VERTICES

    def __init__(self, num_vertices=65):
        self.num_vertices = num_vertices
        
        # Define indices for each anatomical part
        self.body_indices = list(range(0, 23))
        self.lh_indices = list(range(23, 44))
        self.rh_indices = list(range(44, 65))

        # Face vertices, if present, always come after body+hands (65 .. num_vertices-1),
        # matching dataset.py's face concatenation (appended along the VERTEX axis after
        # body/hands). num_vertices > 65 is how this class detects that face is included --
        # matches how queue_train.py's calculate_num_vertices() sets it.
        self.has_face = num_vertices > 65
        if self.has_face:
            expected_face_count = num_vertices - 65
            if expected_face_count != self.NUM_FACE_VERTICES:
                raise ValueError(
                    f"num_vertices={num_vertices} implies {expected_face_count} face vertices, "
                    f"but SkeletonGraph's own face index list has {self.NUM_FACE_VERTICES}. "
                    f"Keep _LIPS_INDICES/_LEFT_EYE_INDICES/etc. here in sync with "
                    f"extract_face_keypoints.py's SELECTED_INDICES."
                )
            self.face_indices = list(range(65, num_vertices))
        else:
            self.face_indices = []
        
        # 1. Decoupled Matrices (for DecoupledSTGCNBlock)
        self.A_body = self._get_subgraph_adjacency(self.body_indices, self._get_body_edges())
        self.A_lh = self._get_subgraph_adjacency(self.lh_indices, self._get_hand_edges(offset=23))
        self.A_rh = self._get_subgraph_adjacency(self.rh_indices, self._get_hand_edges(offset=44))
        if self.has_face:
            self.A_face = self._get_subgraph_adjacency(self.face_indices, self._get_face_edges(offset=65))
        
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

    def _get_face_edges(self, offset):
        """
        Approximate face-mesh connectivity: each region (lips/eyes/eyebrows) is
        connected as a chain following the SAME point order used to build
        SELECTED_INDICES in extract_face_keypoints.py (which traces each
        region's contour). Lips and eyes are closed loops (their contours
        genuinely close on the face); eyebrows are open arcs (not closed).

        NOTE: this is NOT the literal MediaPipe FACEMESH_TESSELATION graph --
        it's a defensible approximation built from the ordering already present
        in these index lists, chosen to avoid guessing at MediaPipe's internal
        triangulation. Good enough for a spatial-locality prior; not a claim of
        anatomical precision beyond "these points are on the same feature".
        """
        pos = {raw_idx: local_pos for local_pos, raw_idx in enumerate(self._FACE_SELECTED_INDICES)}

        def chain(raw_indices, close_loop):
            edges = []
            for k in range(len(raw_indices) - 1):
                edges.append((pos[raw_indices[k]], pos[raw_indices[k + 1]]))
            if close_loop:
                edges.append((pos[raw_indices[-1]], pos[raw_indices[0]]))
            return edges

        edges = []
        edges += chain(self._LIPS_INDICES, close_loop=True)
        edges += chain(self._LEFT_EYE_INDICES, close_loop=True)
        edges += chain(self._RIGHT_EYE_INDICES, close_loop=True)
        edges += chain(self._LEFT_EYEBROW_INDICES, close_loop=False)
        edges += chain(self._RIGHT_EYEBROW_INDICES, close_loop=False)

        return [(i + offset, j + offset) for i, j in edges]

    def _get_all_edges(self):
        """Combines all edges to build the full skeleton graph."""
        edges = self._get_body_edges()
        edges.extend(self._get_hand_edges(offset=23))
        edges.extend(self._get_hand_edges(offset=44))
        # Connect hands to arms
        edges.append((15, 23)) 
        edges.append((16, 44))

        if self.has_face:
            edges.extend(self._get_face_edges(offset=65))
            # Anchor face to body. Body vertex 0 is the nose: BODY_LANDMARKS_KEPT
            # preserves raw MediaPipe Pose indices 0..22 in order (confirmed by
            # cross-checking extract_poses.py's "shoulders at index 11/12" comment
            # and every edge in _get_body_edges against MediaPipe Pose's standard
            # 33-point topology -- all 22 edges match exactly), and index 0 = nose
            # in that topology. Connect nose to both inner eye corners (raw index
            # 133 = left eye inner corner, 362 = right eye inner corner) as two
            # symmetric anchor edges, mirroring how each hand anchors to its wrist
            # above (15->23, 16->44).
            face_pos = {raw_idx: local_pos for local_pos, raw_idx in enumerate(self._FACE_SELECTED_INDICES)}
            left_eye_inner = 65 + face_pos[133]
            right_eye_inner = 65 + face_pos[362]
            edges.append((0, left_eye_inner))
            edges.append((0, right_eye_inner))

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