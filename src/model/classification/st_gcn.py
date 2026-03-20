import torch
import torch.nn as nn
import numpy as np

# ==========================================
# DYNAMIC GRAPH BUILDER
# ==========================================
class DynamicSkeletonGraph:
    """
    Dynamically builds the Adjacency Matrix (A) based on the active landmarks,
    using anatomically exact mappings for hands, pose, and MediaPipe face meshes.
    """
    def __init__(self, active_landmarks, landmark_dims):
        self.active_landmarks = active_landmarks
        self.landmark_dims = landmark_dims
        # Total nodes = Total Features / 2 (since each node has X, Y coordinates)
        self.num_nodes = sum([landmark_dims[lm] // 2 for lm in active_landmarks])
        
        self.A = self._build_adjacency_matrix()

    def _get_internal_edges(self, lm_name):
        """Returns the local edge connections for a specific MediaPipe component."""
        edges = []
        nodes = self.landmark_dims[lm_name] // 2

        if 'hand' in lm_name:
            # Exact MediaPipe 21-point Hand Kinematics
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),           # Index
                (5, 9), (9, 10), (10, 11), (11, 12),      # Middle
                (9, 13), (13, 14), (14, 15), (15, 16),    # Ring
                (13, 17), (17, 18), (18, 19), (19, 20),   # Pinky
                (0, 17)                                   # Palm Closure
            ]
            
        elif 'lips' in lm_name or 'outer' in lm_name or 'inner' in lm_name:
            # Connect the 20 lip points in a continuous loop
            edges = [(i, (i + 1) % nodes) for i in range(nodes)]
            
        elif 'pose' in lm_name:
            # 15-node Pose (MediaPipe 0-16, EXCLUDING 9 and 10)
            # Indices >= 11 are shifted down by 2 to account for missing mouth nodes!
            edges = [
                # --- Face/Head (No shift, indices 0-8) ---
                (0, 1), (1, 2), (2, 3), (3, 7), # Nose to Left Ear
                (0, 4), (4, 5), (5, 6), (6, 8), # Nose to Right Ear
                # --- Torso & Arms (Shifted indices) ---
                (9, 10),                        # Shoulders (Raw 11-12 -> Shifted 9-10)
                (9, 11), (11, 13),              # Left Arm (Raw 11-13-15 -> Shifted 9-11-13)
                (10, 12), (12, 14)              # Right Arm (Raw 12-14-16 -> Shifted 10-12-14)
            ]
            
        elif 'cheeks' in lm_name:
            # 34-node Cheeks (Exact MediaPipe Delaunay Triangulation)
            # Array Indices 0-16: Right Cheek | Indices 17-33: Left Cheek
            def add_exact_mediapipe_patch(start_idx):
                local_edges = [
                    # Horizontal Rows
                    (0,1), (1,2), (2,3),             # Row 1
                    (4,5), (5,6), (6,7),             # Row 2
                    (8,9), (9,10), (10,11), (11,12), # Row 3
                    (13,14), (14,15), (15,16),       # Row 4
                    # Row 1 down to Row 2
                    (0,4), (0,5), (1,5), (1,6), (2,6), (3,6), (3,7),
                    # Row 2 down to Row 3
                    (4,8), (4,9), (5,9), (5,10), (6,10), (6,11), (7,11), (7,12),
                    # Row 3 down to Row 4
                    (9,13), (10,13), (10,14), (10,15), (11,15), (11,16), (12,16)
                ]
                # Apply offset for left/right symmetry
                for u, v in local_edges:
                    edges.append((start_idx + u, start_idx + v))

            # Build Right Cheek Mesh (Starts at index 0)
            add_exact_mediapipe_patch(0)
            # Build Left Cheek Mesh (Starts at index 17)
            add_exact_mediapipe_patch(17)
            
        return edges

    def _build_adjacency_matrix(self):
        A = np.zeros((self.num_nodes, self.num_nodes))
        
        current_offset = 0
        for lm in self.active_landmarks:
            num_local_nodes = self.landmark_dims[lm] // 2
            local_edges = self._get_internal_edges(lm)
            
            # Add edges to the global matrix, shifted by the current offset
            for i, j in local_edges:
                global_i = i + current_offset
                global_j = j + current_offset
                A[global_i, global_j] = 1
                A[global_j, global_i] = 1 # Undirected graph
                
            current_offset += num_local_nodes
            
        # Add Self-Loops (Identity Matrix) so nodes remember their own features
        A = A + np.eye(self.num_nodes)
        
        # Spectral Normalization: D^{-1/2} A D^{-1/2}
        row_sum = A.sum(axis=1)
        D_inv_sqrt = np.power(row_sum, -0.5).flatten()
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
        D_mat_inv_sqrt = np.diag(D_inv_sqrt)
        A_normalized = D_mat_inv_sqrt @ A @ D_mat_inv_sqrt
        
        return torch.tensor(A_normalized, dtype=torch.float32)

# ==========================================
# ST-GCN BLOCK
# ==========================================
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, temporal_kernel_size=9, stride=1, residual=True):
        super().__init__()
        self.register_buffer('A', A) # Moves A to GPU automatically
        
        # Spatial Graph Convolution (Mixes node features)
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Temporal Convolution (Mixes time frames)
        padding = ((temporal_kernel_size - 1) // 2, 0)
        self.temporal_conv = nn.Conv2d(
            out_channels, out_channels, 
            kernel_size=(temporal_kernel_size, 1), 
            stride=(stride, 1), 
            padding=padding
        )
        
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual connection to prevent vanishing gradients
        if not residual:
            self.residual = lambda x: 0
        elif in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = lambda x: x

    def forward(self, x):
        res = self.residual(x)
        
        # 1. Spatial Forward: Multiply features by Adjacency Matrix
        # (Batch, Channels, Time, Nodes_V) x (Nodes_V, Nodes_W) -> (B, C, T, W)
        x = torch.einsum('nctv,vw->nctw', (x, self.A))
        x = self.spatial_conv(x)
        
        # 2. Temporal Forward
        x = self.temporal_conv(x)
        x = self.batch_norm(x)
        
        return self.relu(x + res)

# ==========================================
# FULL ST-GCN CLASSIFIER
# ==========================================
class STGCN_Classifier(nn.Module):
    def __init__(self, active_landmarks, landmark_dims, num_classes=4, in_channels=2, dropout_prob=0.5):
        """
        in_channels: 2 (X and Y coordinates)
        """
        super().__init__()
        
        # Build the graph dynamically
        self.graph = DynamicSkeletonGraph(active_landmarks, landmark_dims)
        A = self.graph.A
        
        # Network Architecture
        self.data_bn = nn.BatchNorm1d(in_channels * self.graph.num_nodes)
        
        self.st_gcn_networks = nn.ModuleList([
            STGCNBlock(in_channels, 64, A, residual=False),
            STGCNBlock(64, 64, A),
            STGCNBlock(64, 128, A, stride=2), # Downsample time
            STGCNBlock(128, 128, A),
            STGCNBlock(128, 256, A, stride=2), # Downsample time again
            STGCNBlock(256, 256, A)
        ])
        
        self.fc = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(128, num_classes)
            )

    def forward(self, x, lengths=None):
        B, T, F = x.shape
        V = self.graph.num_nodes
        C = 2 # X, Y
        
        # --- DATA RESHAPING ---
        x = x.view(B, T, V, C)
        x = x.permute(0, 3, 1, 2).contiguous() # -> [B, C, T, V]
        
        N, C_dim, T_dim, V_dim = x.size()
        x = x.permute(0, 1, 3, 2).contiguous().view(N, C_dim * V_dim, T_dim)
        x = self.data_bn(x)
        x = x.view(N, C_dim, V_dim, T_dim).permute(0, 1, 3, 2).contiguous()
        
        # --- ST-GCN FORWARD PASS ---
        for gcn in self.st_gcn_networks:
            x = gcn(x)
            
        # ==========================================
        # DYNAMIC TEMPORAL MASKING (WITH STRIDE FIX)
        # ==========================================
        if lengths is not None:
            # 1. Shrink lengths to match the two stride=2 convolutions
            # Math formula for Conv1D stride=2 with same padding is: ceil(length / 2)
            out_lengths = lengths.clone()
            out_lengths = (out_lengths + 1) // 2  # Shrink from block 3
            out_lengths = (out_lengths + 1) // 2  # Shrink from block 5
            
            # Get the new, shrunken time dimension of the tensor
            T_out = x.size(2) 
            
            # 2. Create the mask using the NEW lengths and NEW time dimension
            time_steps = torch.arange(T_out, device=x.device).unsqueeze(0) # [1, T_out]
            mask = time_steps < out_lengths.unsqueeze(1) # [B, T_out]
            
            # 3. Reshape mask to broadcast: [B, 1, T_out, 1]
            mask = mask.unsqueeze(1).unsqueeze(-1).float()
            
            # 4. Wipe out padded frames
            x = x * mask 
            
            # 5. Sum the valid features
            x_sum = torch.sum(x, dim=(2, 3)) # -> [B, Channels]
            
            # 6. Divide only by the actual number of valid items
            valid_elements = (out_lengths * V_dim).unsqueeze(1).float() # [B, 1]
            x_pooled = x_sum / (valid_elements + 1e-6) 
            
        else:
            x_pooled = torch.mean(x, dim=(2, 3))
            
        # --- CLASSIFICATION HEAD ---
        return self.fc(x_pooled)