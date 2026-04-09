import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np

from src.data.jsl_dataset import MultiVideoFaceBodySequenceDataset, BucketedBatchSampler, collate_fn_face_body_sequence
from src.model.classification.st_gcn import STGCN_Classifier
from src.utils.metrics import plot_loss_curve

class SubsetWithSamples(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.samples = [dataset.samples[i] for i in indices]

def main():
    parser = argparse.ArgumentParser(description="Train ST-GCN Binary Localization Model")
    parser.add_argument("--data_dir", type=str, default="./data/processed_annotations")
    parser.add_argument("--landmarks", nargs='*', type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0005) # Lowered for GCN stability
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    LANDMARK_DIMS = {
        'lm_face_outer_lips': 40, 'lm_face_inner_lips': 40, 'lm_face_cheeks': 68,
        'lm_body_outer_lips': 40, 'lm_body_inner_lips': 40, 'lm_body_cheeks': 68,
        'lm_pose': 30, 'lm_l_hand': 42, 'lm_r_hand': 42
    }

    lm_str = "-".join([lm.replace('lm_', '') for lm in args.landmarks])
    exp_name = f"STGCN_LOC_{lm_str}"
    print(f"\n[INFO] Starting Binary Localization Experiment: {exp_name}\n")

    # 1. LOAD LOCALIZATION DATASET (Variable Lengths on-the-go)
    full_dataset = MultiVideoFaceBodySequenceDataset(
        root_dir=args.data_dir,
        task='localization',         # <-- Explicitly targets binary segments
        mode='no_img',               
        return_landmarks=True,
        min_seq_len=6,               # Allowing short segments
        max_seq_len=89,
        downsample=1                 # Highly recommended to balance the 0s and 1s!
    )

    train_size = int(0.8 * len(full_dataset))
    indices = torch.randperm(len(full_dataset)).tolist()
    train_dataset = SubsetWithSamples(full_dataset, indices[:train_size])
    valid_dataset = SubsetWithSamples(full_dataset, indices[train_size:])

    train_loader = DataLoader(train_dataset, batch_sampler=BucketedBatchSampler(train_dataset, args.batch_size), collate_fn=collate_fn_face_body_sequence)
    valid_loader = DataLoader(valid_dataset, batch_sampler=BucketedBatchSampler(valid_dataset, args.batch_size), collate_fn=collate_fn_face_body_sequence)

    # 2. INITIALIZE BINARY MODEL
    model = STGCN_Classifier(
        active_landmarks=args.landmarks,
        landmark_dims=LANDMARK_DIMS,
        num_classes=2,               # <-- Binary classes (0: No Movement, 1: Movement)
        in_channels=2,
        dropout_prob=0.5
    ).to(device)

    # 3. CLASS WEIGHTS (Smoothed to prevent gradient explosion)
    class_counts = torch.zeros(2) 
    for sample in train_dataset.samples:
        lbl = sample['label']
        if isinstance(lbl, str): lbl = full_dataset.label_mapping[lbl]
        elif torch.is_tensor(lbl): lbl = lbl.item()
        elif isinstance(lbl, (list, tuple, np.ndarray)): lbl = lbl[0] 
        class_counts[int(lbl)] += 1
        
    print(f"[INFO] Train Set Class Counts (0: None, 1: Move): {class_counts.tolist()}")
    weights = 1.0 / torch.sqrt(class_counts + 1e-6)
    weights = weights / weights.min()
    weights = torch.clamp(weights, max=5.0) 
    
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    
    # Optimizer with Weight Decay & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    def prepare_batch(batch, is_training=False):
        lengths = batch['lengths'].to(device)
        labels = batch['labels'].to(device)
        lm_parts = []
        for key in args.landmarks:
            if key in batch:
                part = batch[key].to(device)
                lm_parts.append(part.view(part.size(0), part.size(1), -1))
            else:
                raise KeyError(f"Landmark {key} missing!")
        
        l_seq = torch.cat(lm_parts, dim=-1) 
        
        # --- KINEMATIC AUGMENTATION ---
        if is_training:
            # Add tiny random coordinate noise to prevent memorization
            l_seq = l_seq + (torch.randn_like(l_seq) * 0.005)
            # Random global scaling
            scale = torch.empty(l_seq.size(0), 1, 1).uniform_(0.95, 1.05).to(device)
            l_seq = l_seq * scale
            
        return l_seq, lengths, labels

    # 4. TRAINING LOOP
    best_f1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            l_seq, lengths, labels = prepare_batch(batch, is_training=True)
            optimizer.zero_grad()
            
            outputs = model(l_seq, lengths) 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)

        # 5. VALIDATION (With Precision/Recall Tracking)
        model.eval()
        total_valid_loss = 0.0
        tp, fp, fn, tn = 0, 0, 0, 0

        with torch.no_grad():
            for batch in valid_loader:
                l_seq, lengths, labels = prepare_batch(batch, is_training=False)
                outputs = model(l_seq, lengths)
                loss = criterion(outputs, labels)
                total_valid_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                
                # Calculate metrics for Class 1 (Movement)
                tp += ((predicted == 1) & (labels == 1)).sum().item()
                fp += ((predicted == 1) & (labels == 0)).sum().item()
                fn += ((predicted == 0) & (labels == 1)).sum().item()
                tn += ((predicted == 0) & (labels == 0)).sum().item()

        valid_loss = total_valid_loss / len(valid_loader)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0

        print(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f}")
        print(f"         Acc: {accuracy*100:.1f}% | Prec: {precision:.3f} | Rec: {recall:.3f} | F1: {f1_score:.3f}")
        
        # Tell scheduler to monitor validation loss
        scheduler.step(valid_loss)
        
        # Save model based on F1 Score
        if f1_score > best_f1:
            best_f1 = f1_score
            torch.save(model.state_dict(), os.path.join('saved_models', f"model_{exp_name}.pth"))

    print(f"\nTraining Complete! Best Validation F1: {best_f1:.3f}")

if __name__ == "__main__":
    main()