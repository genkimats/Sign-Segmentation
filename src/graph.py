import numpy as np

class SkeletonGraph:
    def __init__(self, num_vertices=65):
        self.num_vertices = num_vertices
        self.edges = self._get_edges()
        self.A = self._get_adjacency_matrix()

    def _get_edges(self):
        edges = []
        
        # 1. Body Edges (Assuming MediaPipe upper-body 0 to 22)
        # Note: Adjust these if your 23 custom body landmarks map differently!
        body_edges = [
            (0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8), # Face
            (9,10), (11,12), # Shoulders/Mouth
            (11,13), (13,15), (15,17), (15,19), (15,21), (17,19), # Left Arm
            (12,14), (14,16), (16,18), (16,20), (16,22), (18,20)  # Right Arm
        ]
        edges.extend(body_edges)
        
        # 2. Left Hand Edges (Vertices 23 to 43)
        offset_lh = 23
        hand_edges = [
            (0,1), (1,2), (2,3), (3,4),       # Thumb
            (0,5), (5,6), (6,7), (7,8),       # Index
            (0,9), (9,10), (10,11), (11,12),  # Middle
            (0,13), (13,14), (14,15), (15,16),# Ring
            (0,17), (17,18), (18,19), (19,20) # Pinky
        ]
        edges.extend([(i + offset_lh, j + offset_lh) for i, j in hand_edges])
        
        # 3. Right Hand Edges (Vertices 44 to 64)
        offset_rh = 44
        edges.extend([(i + offset_rh, j + offset_rh) for i, j in hand_edges])
        
        # 4. Connect Arms to Hands
        # MediaPipe Left Wrist is 15. Hand root is 23.
        edges.append((15, 23)) 
        # MediaPipe Right Wrist is 16. Hand root is 44.
        edges.append((16, 44))
        
        return edges

    def _get_adjacency_matrix(self):
        """Creates a normalized adjacency matrix with self-loops."""
        A = np.zeros((self.num_vertices, self.num_vertices))
        for i, j in self.edges:
            # Ensure indices don't go out of bounds if custom layouts differ
            if i < self.num_vertices and j < self.num_vertices:
                A[i, j] = 1
                A[j, i] = 1
            
        # Add self-loops (identity matrix) so a joint remembers its own position
        A = A + np.eye(self.num_vertices)
        
        # Normalize the matrix to prevent gradient explosion during message passing
        D = np.diag(np.sum(A, axis=1) ** -0.5)
        A_normalized = D @ A @ D
        return A_normalized