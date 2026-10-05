import numpy as np

from src.face_subsets import (FACE_REGIONS, ANCHOR_LEFT_EYE_INNER, ANCHOR_RIGHT_EYE_INNER,
                              LIPS_INDICES, LEFT_EYE_INDICES, RIGHT_EYE_INDICES,
                              face_subset_for_vertex_count, face_subset_raw_indices)


class SkeletonGraph:
    """
    Body (23) + left hand (21) + right hand (21) = 65 vertices, optionally followed by
    face vertices. The face part can be any subset defined in src/face_subsets.py
    ("full" 83, "compact" 31, "eyes_brows" 22, "minimal" 18); the subset is inferred
    from num_vertices - 65 (subset sizes are unique), so callers only pass num_vertices.

    face_only=True: the graph contains ONLY face vertices (0 .. num_vertices-1), no body or
    hands; the subset is inferred from num_vertices itself. Decoupled body/hand matrices
    don't exist in this mode.
    """

    def __init__(self, num_vertices=65, face_only=False):
        self.num_vertices = num_vertices
        self.face_only = face_only
        if face_only:
            self.body_indices, self.lh_indices, self.rh_indices = [], [], []
            self.has_face = True
            self.face_offset = 0
            self.face_subset = face_subset_for_vertex_count(num_vertices)
            self.face_raw_indices = face_subset_raw_indices(self.face_subset)
            self.face_indices = list(range(0, num_vertices))
            self.A_face = self._get_subgraph_adjacency(self.face_indices, self._get_face_edges(offset=0))
            self.A = self._get_subgraph_adjacency(self.face_indices, self._get_all_edges())
            return
        self.face_offset = 65
        
        # Define indices for each anatomical part
        self.body_indices = list(range(0, 23))
        self.lh_indices = list(range(23, 44))
        self.rh_indices = list(range(44, 65))

        # Face vertices, if present, always come after body+hands (65 .. num_vertices-1),
        # matching dataset.py's face concatenation (appended along the VERTEX axis).
        self.has_face = num_vertices > 65
        if self.has_face:
            self.face_subset = face_subset_for_vertex_count(num_vertices - 65)
            self.face_raw_indices = face_subset_raw_indices(self.face_subset)  # vertex order
            self.face_indices = list(range(65, num_vertices))
        else:
            self.face_subset = None
            self.face_raw_indices = []
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

    def _face_local_pos(self):
        """raw MediaPipe index -> local face vertex position, for the loaded subset."""
        return {raw: i for i, raw in enumerate(self.face_raw_indices)}

    def _get_face_edges(self, offset):
        """
        Approximate face connectivity: each region (lips / eyes / eyebrows) is a chain
        following its contour order, keeping only the points in the loaded subset
        (removed points are skipped, so their neighbours connect directly). Lips and
        eyes are closed loops, eyebrows open arcs. A spatial-locality prior, not the
        literal MediaPipe tessellation.
        """
        pos = self._face_local_pos()
        edges = []
        for raw_list, close_loop in FACE_REGIONS:
            kept = []
            for raw in raw_list:
                if raw in pos and (not kept or kept[-1] != raw):
                    kept.append(raw)
            if len(kept) > 1 and kept[0] == kept[-1]:
                kept = kept[:-1]  # contour list repeats its first point
            for k in range(len(kept) - 1):
                edges.append((pos[kept[k]], pos[kept[k + 1]]))
            if close_loop and len(kept) > 2:
                edges.append((pos[kept[-1]], pos[kept[0]]))
        edges = sorted(set(tuple(sorted(e)) for e in edges if e[0] != e[1]))
        return [(i + offset, j + offset) for i, j in edges]

    def _get_all_edges(self):
        """Combines all edges to build the full skeleton graph."""
        if self.face_only:
            return self._get_face_edges(offset=0)
        edges = self._get_body_edges()
        edges.extend(self._get_hand_edges(offset=23))
        edges.extend(self._get_hand_edges(offset=44))
        # Connect hands to arms
        edges.append((15, 23)) 
        edges.append((16, 44))

        if self.has_face:
            edges.extend(self._get_face_edges(offset=65))
            # Anchor face to body: nose (body vertex 0) -> both inner eye corners
            # (raw 133 / 362; every subset keeps them), mirroring how each hand
            # anchors to its wrist above (15->23, 16->44).
            pos = self._face_local_pos()
            edges.append((0, 65 + pos[ANCHOR_LEFT_EYE_INNER]))
            edges.append((0, 65 + pos[ANCHOR_RIGHT_EYE_INNER]))

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

    # ==========================================================================
    # HD-GCN support: hierarchical (multi-hop) adjacency decomposition.
    # ==========================================================================
    def _all_pairs_hop_distance(self, node_indices, edges, max_hop):
        """
        BFS shortest-path distance (in hops) between every pair of vertices,
        using ONLY the given edges (1-hop = a direct edge). Distances beyond
        max_hop are left as "infinite" (never selected for any level).
        """
        num_nodes = len(node_indices)
        idx_map = {g: l for l, g in enumerate(node_indices)}

        raw_adj = np.zeros((num_nodes, num_nodes), dtype=np.int32)
        for i, j in edges:
            if i in idx_map and j in idx_map:
                li, lj = idx_map[i], idx_map[j]
                raw_adj[li, lj] = 1
                raw_adj[lj, li] = 1

        INF = max_hop + 1
        dist = np.full((num_nodes, num_nodes), INF, dtype=np.int32)
        for src in range(num_nodes):
            dist[src, src] = 0
            frontier = [src]
            visited = {src}
            d = 0
            while frontier and d < max_hop:
                d += 1
                next_frontier = []
                for u in frontier:
                    for v in np.where(raw_adj[u] > 0)[0]:
                        v = int(v)
                        if v not in visited:
                            visited.add(v)
                            dist[src, v] = d
                            next_frontier.append(v)
                frontier = next_frontier

        return dist

    def get_hop_adjacencies(self, max_hop=3):
        """
        HD-GCN's core idea (Lee et al., ICCV 2023): instead of one adjacency
        covering all neighbor distances, decompose the graph into separate
        hop-distance LEVELS -- level h contains an edge between i and j iff
        their shortest-path distance is EXACTLY h -- so direct neighbors,
        2-hop neighbors, 3-hop neighbors etc. each get their own dedicated
        (normalized) adjacency matrix instead of being flattened into one.

        Returns a list of `max_hop` normalized (V, V) matrices, covering the
        UNIFIED graph (same vertex set as self.A -- body+hands, plus face
        when present).
        """
        node_indices = list(range(self.num_vertices))
        edges = self._get_all_edges()
        dist = self._all_pairs_hop_distance(node_indices, edges, max_hop)

        num_nodes = self.num_vertices
        matrices = []
        for h in range(1, max_hop + 1):
            H = (dist == h).astype(np.float64)
            H = H + np.eye(num_nodes)  # self-loops, same as _get_subgraph_adjacency
            D = np.diag(np.sum(H, axis=1) ** -0.5)
            matrices.append(D @ H @ D)
        return matrices

    # ==========================================================================
    # HyperSign support: anatomically-grounded hyperedges (multi-vertex groups).
    # ==========================================================================
    def get_anatomical_hyperedges(self):
        """
        Explicit, hand-designed multi-vertex groups -- a simplification of
        HyperSign's k-NN-constructed "dynamic geometric hypergraphs" (we use
        fixed, interpretable anatomical groups instead of a differentiable
        k-NN construction). Each group is a hand-shape or coordination unit
        that's naturally a MULTI-way relationship, not a pairwise one -- e.g.
        all 5 fingertips jointly define hand aperture/shape in a way no
        single pairwise edge captures.

        Returns a list of vertex-index lists (each list = one hyperedge).
        """
        hyperedges = []
        if self.face_only:
            face_pos = self._face_local_pos()
            for raw_list in (LIPS_INDICES, LEFT_EYE_INDICES, RIGHT_EYE_INDICES):
                group = [face_pos[i] for i in raw_list if i in face_pos]
                if len(set(group)) >= 2:
                    hyperedges.append(group)
            return hyperedges

        # Hand topology (within each 21-point hand block, LOCAL indices):
        # 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
        # (standard MediaPipe Hands numbering -- matches _get_hand_edges above).
        fingertip_local = [4, 8, 12, 16, 20]
        for offset in (23, 44):  # left hand, right hand
            tips = [offset + i for i in fingertip_local]
            hyperedges.append(tips)                    # all 5 fingertips: hand shape/aperture
            hyperedges.append([offset] + tips)          # + wrist: shape relative to hand root

        # Upper-body configuration: both shoulders, elbows, wrists together
        # (raw MediaPipe Pose indices 11-16, confirmed against the body edge
        # list and extract_poses.py's shoulder comment -- see graph.py history).
        hyperedges.append([11, 12, 13, 14, 15, 16])

        if self.has_face:
            face_pos = self._face_local_pos()

            def face_group(raw_indices):
                # Same construction as before (contour order, including the repeated lips
                # point), so the "full" subset reproduces the original hyperedges exactly.
                return [65 + face_pos[i] for i in raw_indices if i in face_pos]

            for group in (face_group(LIPS_INDICES), face_group(LEFT_EYE_INDICES),
                          face_group(RIGHT_EYE_INDICES)):
                if len(set(group)) >= 2:
                    hyperedges.append(group)

        return hyperedges