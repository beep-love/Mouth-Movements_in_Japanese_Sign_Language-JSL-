import torch
import torch.nn as nn
import torchvision.models as models

class Attention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.score = nn.Linear(input_dim, 1)

    def forward(self, rnn_out, lengths, return_weights=False):
        """
        rnn_out: (B, T, H)
        lengths: (B,)
        Returns: (B, H) attention-weighted sum
        """
        B, T, H = rnn_out.size()
        device = rnn_out.device
        # Mask to ignore padded positions
        mask = torch.arange(T, device=device)[None, :] < lengths[:, None] # (B, T)

        scores = self.score(rnn_out).squeeze(-1) # (B, T)
        scores[~mask] = -float('inf') # Mask out padded positions
        weights = torch.softmax(scores, dim=1) # (B, T)

        weighted_sum = (rnn_out * weights.unsqueeze(-1)).sum(dim=1)

        if return_weights: return weighted_sum, weights  # (B, H), (B, T)
        return weighted_sum

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.conv(x))

class CustomCNN(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout(0.2), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), ResidualBlock(64), nn.Dropout(0.3), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), ResidualBlock(128), nn.Dropout(0.4), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveMaxPool2d((1, 1))
        )
        self.fc = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ImageBranch(nn.Module):
    """Handles either Face or Body image sequences."""
    def __init__(self, backbone_type='resnet18', cnn_out_dim=128, rnn_type='gru', rnn_hidden_dim=128):
        super().__init__()
        self.backbone_type = backbone_type
        
        # 1. Spatial Backbone
        if backbone_type == 'resnet18':
            resnet = models.resnet18(pretrained=True)
            self.cnn = nn.Sequential(*list(resnet.children())[:-1])
            self.cnn_fc = nn.Linear(512, cnn_out_dim)
        elif backbone_type == 'custom':
            self.cnn = CustomCNN(output_dim=cnn_out_dim)
        else:
            raise ValueError(f"Unknown backbone: {backbone_type}")

        # 2. Temporal Backbone
        rnn_class = nn.LSTM if rnn_type.lower() == 'lstm' else nn.GRU
        self.rnn = rnn_class(input_size=cnn_out_dim, hidden_size=rnn_hidden_dim, batch_first=True, bidirectional=True)
        
        # 3. Attention
        self.attn = Attention(rnn_hidden_dim * 2)

    def forward(self, x_seq, lengths):
        B, T, C, H, W = x_seq.shape
        x_flat = x_seq.view(B * T, C, H, W)

        if self.backbone_type == 'resnet18':
            feats = self.cnn(x_flat).view(B * T, -1)
            feats = self.cnn_fc(feats)
        else:
            feats = self.cnn(x_flat)

        feats = feats.view(B, T, -1)
        
        packed = nn.utils.rnn.pack_padded_sequence(feats, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        
        return self.attn(out, lengths)

class LandmarkBranch(nn.Module):
    """Handles normalized numerical landmarks."""
    def __init__(self, input_dim, rnn_type='gru', rnn_hidden_dim=64, dropout_prob=0.2):
        super().__init__()
        
        # 1. Spatial MLP (Expand-then-Compress Strategy)
        # 256 gives plenty of room for the 226-dim full body input to interact
        self.mlp = nn.Sequential(
            # --- EXPAND ---
            nn.Linear(input_dim, 512), 
            nn.BatchNorm1d(512),      
            nn.ReLU(), 
            nn.Dropout(dropout_prob),

            # --- COMPRESS PHASE 1 ---
            nn.Linear(512, 256), 
            nn.BatchNorm1d(256),      
            nn.ReLU(), 
            nn.Dropout(dropout_prob),

            # --- COMPRESS PHASE 2 ---
            nn.Linear(256, 128), 
            nn.BatchNorm1d(128), 
            nn.ReLU(), 
            nn.Dropout(dropout_prob),
            
            nn.Linear(128, 64)        
        )
        
        # 2. Temporal Backbone
        rnn_class = nn.LSTM if rnn_type.lower() == 'lstm' else nn.GRU
        self.rnn = rnn_class(input_size=64, hidden_size=rnn_hidden_dim, batch_first=True, bidirectional=True)
        
        # 3. Attention
        self.attn = Attention(rnn_hidden_dim * 2)

    def forward(self, lm_seq, lengths):
        B, T, F = lm_seq.shape
        
        # BatchNorm1d expects (Batch, Features), so we flatten the time dimension temporarily
        lm_flat = lm_seq.view(B * T, F)
        lm_feat = self.mlp(lm_flat)
        lm_feat = lm_feat.view(B, T, -1)
        
        packed = nn.utils.rnn.pack_padded_sequence(lm_feat, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        
        return self.attn(out, lengths)

class MultimodalSequenceClassifier(nn.Module):
    def __init__(self, mode='both', use_landmarks=True, lm_input_dim=None,
                 face_backbone='custom', body_backbone='resnet18', rnn_type='gru', 
                 cnn_out_dim=128, rnn_hidden_dim=128, lm_rnn_hidden=64,
                 num_classes=4, dropout_prob=0.3):
        """
        The Ultimate Modular Classifier.
        Args:
            mode (str): 'face', 'body', 'both', or 'no_img'.
            use_landmarks (bool): Whether to process the landmark vectors.
            lm_input_dim (int): Required if use_landmarks=True. Total flattened coordinates per frame.
            face_backbone (str): Backbone for 112x112 face crops ('custom' or 'resnet18').
            body_backbone (str): Backbone for 224x224 body crops ('custom' or 'resnet18').
            rnn_type (str): 'lstm' or 'gru'.
        """
        super().__init__()
        assert mode in ['face', 'body', 'both', 'no_img']
        
        self.mode = mode
        self.use_landmarks = use_landmarks
        
        # Build Branches Dynamically
        final_concat_dim = 0
        
        if self.mode in ['face', 'both']:
            print(f"[MODEL] Initializing FACE Branch ({face_backbone} + Bi-{rnn_type.upper()})")
            # Pass the specific face_backbone here!
            self.face_branch = ImageBranch(face_backbone, cnn_out_dim, rnn_type, rnn_hidden_dim)
            final_concat_dim += (rnn_hidden_dim * 2)
            
        if self.mode in ['body', 'both']:
            print(f"[MODEL] Initializing BODY Branch ({body_backbone} + Bi-{rnn_type.upper()})")
            # Pass the specific body_backbone here!
            self.body_branch = ImageBranch(body_backbone, cnn_out_dim, rnn_type, rnn_hidden_dim)
            final_concat_dim += (rnn_hidden_dim * 2)
            
        if self.use_landmarks:
            assert lm_input_dim is not None, "You must provide lm_input_dim if use_landmarks is True!"
            print(f"[MODEL] Initializing LANDMARK Branch (MLP + Bi-{rnn_type.upper()})")
            self.lm_branch = LandmarkBranch(lm_input_dim, rnn_type, lm_rnn_hidden, dropout_prob)
            final_concat_dim += (lm_rnn_hidden * 2)

        if final_concat_dim == 0:
            raise ValueError("Model has no active branches based on mode and use_landmarks!")

        # Dynamic Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(final_concat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, num_classes)
        )

    def forward(self, lengths, face_seq=None, body_seq=None, lm_seq=None):
        """
        Pass in the sequences we have. If a branch is inactive, it is ignored.
        """
        features = []

        if self.mode in ['face', 'both'] and face_seq is not None:
            features.append(self.face_branch(face_seq, lengths))
            
        if self.mode in ['body', 'both'] and body_seq is not None:
            features.append(self.body_branch(body_seq, lengths))
            
        if self.use_landmarks and lm_seq is not None:
            features.append(self.lm_branch(lm_seq, lengths))

        # Fuse all active modality vectors together
        combined = torch.cat(features, dim=1) 
        
        return self.classifier(combined)