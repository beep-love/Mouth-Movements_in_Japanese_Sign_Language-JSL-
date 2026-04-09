import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np

# Imports from our modular structure
from src.data.jsl_dataset import MultiVideoFaceBodySequenceDataset, BucketedBatchSampler, collate_fn_face_body_sequence
from src.model.classification.st_gcn import STGCN_Classifier
from src.utils.metrics import plot_loss_curve, plot_confusion_matrix

class SubsetWithSamples(Subset):
    """Wrapper to ensure the batch sampler can still access the 'samples' list for length grouping."""
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.samples = [dataset.samples[i] for i in indices]

def main():
    parser = argparse.ArgumentParser(description="Train ST-GCN Kinematic Model")
    parser.add_argument("--data_dir", type=str, default="./data/processed_annotations", help="Path to processed dataset")
    
    # Dynamic Landmark List for Ablation Studies
    parser.add_argument(
        "--landmarks", 
        nargs='*', 
        type=str, 
        required=True, 
        help="List of landmarks to include (e.g. --landmarks lm_face_inner_lips lm_pose)"
    )
    
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0003)      # Slightly higher LR is safe for GCNs
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # DYNAMIC LANDMARK CONFIGURATION
    # ---------------------------------------------------------
    LANDMARK_DIMS = {
        'lm_face_outer_lips': 40,   # 20 nodes * 2
        'lm_face_inner_lips': 40,   # 20 nodes * 2
        'lm_face_cheeks': 68,       # 34 nodes * 2 (Triangulated Patch)
        'lm_body_outer_lips': 40,   
        'lm_body_inner_lips': 40,   
        'lm_body_cheeks': 68,       
        'lm_pose': 30,              # 15 nodes * 2
        'lm_l_hand': 42,            # 21 nodes * 2
        'lm_r_hand': 42             # 21 nodes * 2
    }

    # Validate requested landmarks
    for lm in args.landmarks:
        if lm not in LANDMARK_DIMS:
            raise ValueError(f"Unknown landmark key '{lm}'. Available: {list(LANDMARK_DIMS.keys())}")
            
    lm_str = "-".join([lm.replace('lm_', '') for lm in args.landmarks])
    exp_name = f"STGCN_{lm_str}"
    
    print(f"\n[INFO] Active Landmarks: {args.landmarks}")
    print(f"[INFO] Starting Experiment: {exp_name}\n")

    # ---------------------------------------------------------
    # DATASET & DATALOADERS (Strictly no_img mode for speed!)
    # ---------------------------------------------------------
    full_dataset = MultiVideoFaceBodySequenceDataset(
        root_dir=args.data_dir,
        task='classification',
        mode='no_img',               # <-- Forces blazing fast coordinate-only loading
        return_landmarks=True,
        min_seq_len=10,
        max_seq_len=89,
        downsample=1 
    )

    train_size = int(0.8 * len(full_dataset))
    indices = torch.randperm(len(full_dataset)).tolist()
    train_dataset = SubsetWithSamples(full_dataset, indices[:train_size])
    valid_dataset = SubsetWithSamples(full_dataset, indices[train_size:])

    print(f"\nDataset -> Total: {len(full_dataset)} | Train: {len(train_dataset)} | Val: {len(valid_dataset)}\n")

    train_loader = DataLoader(train_dataset, batch_sampler=BucketedBatchSampler(train_dataset, args.batch_size), collate_fn=collate_fn_face_body_sequence)
    valid_loader = DataLoader(valid_dataset, batch_sampler=BucketedBatchSampler(valid_dataset, args.batch_size), collate_fn=collate_fn_face_body_sequence)

    # ---------------------------------------------------------
    # MODEL INITIALIZATION
    # ---------------------------------------------------------
    model = STGCN_Classifier(
        active_landmarks=args.landmarks,
        landmark_dims=LANDMARK_DIMS,
        num_classes=4,
        in_channels=2,
        dropout_prob=0.5
    ).to(device)

    # ---------------------------------------------------------
    # DYNAMIC CLASS WEIGHTS (Protecting the minority classes)
    # ---------------------------------------------------------
    class_counts = torch.zeros(4) 
    for sample in train_dataset.samples:
        lbl = sample['label']
        if isinstance(lbl, str):
            lbl = full_dataset.label_mapping[lbl]
        elif torch.is_tensor(lbl):
            lbl = lbl.item()
        elif isinstance(lbl, (list, tuple, np.ndarray)):
            lbl = lbl[0] 
        class_counts[int(lbl)] += 1
        
    print(f"\n[INFO] Train Set Class Counts: {class_counts.tolist()}")
    weights = 1.0 / torch.sqrt(class_counts + 1e-6)
    weights = weights / weights.min()

    # Clamp extreme weights to a reasonable range (e.g., max weight of 10) to prevent instability
    # weights = torch.clamp(weights, max=10.0)

    print(f"[INFO] Applied Class Weights: {weights.tolist()}\n")

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # The Scheduler :- This will cut the LR by 50% (factor=0.5) if the validation loss doesn't improve for 5 epochs (patience=5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    epoch_train_losses, epoch_valid_losses, epoch_accuracies = [], [], []
    best_acc = 0.0
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    # ---------------------------------------------------------
    # BATCH HELPER (Coordinate Extraction Only)
    # ---------------------------------------------------------
    def prepare_batch(batch, is_training=False):
        lengths = batch['lengths'].to(device)
        labels = batch['labels'].to(device)
        
        lm_parts = []
        for key in args.landmarks: 
            if key in batch:
                part = batch[key].to(device)
                lm_parts.append(part.view(part.size(0), part.size(1), -1))
            else:
                raise KeyError(f"Requested landmark '{key}' missing from batch!")
                
        # Concat into [B, T, Total_Features]
        l_seq = torch.cat(lm_parts, dim=-1) 

        # --- KINEMATIC AUGMENTATION ---
        if is_training:
            # Coordinate Jitter: Add tiny random noise to coordinates
            # This simulates slight tracking errors from MediaPipe and prevents memorization
            noise = torch.randn_like(l_seq) * 0.005  # 0.5% screen width variance
            l_seq = l_seq + noise
            
            # Random Global Scaling: Make the whole face/body 5% bigger or smaller
            scale = torch.empty(l_seq.size(0), 1, 1).uniform_(0.95, 1.05).to(device)
            l_seq = l_seq * scale
        return l_seq, lengths, labels

    # ---------------------------------------------------------
    # TRAINING LOOP
    # ---------------------------------------------------------
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            l_seq, lengths, labels = prepare_batch(batch, is_training=True)

            optimizer.zero_grad()
            # ST-GCN forward pass takes (x, lengths)
            outputs = model(l_seq, lengths) 
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        epoch_train_losses.append(train_loss)

        # Validation
        model.eval()
        total_valid_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for batch in valid_loader:
                l_seq, lengths, labels = prepare_batch(batch, is_training=False)
                
                outputs = model(l_seq, lengths)
                loss = criterion(outputs, labels)
                total_valid_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        valid_loss = total_valid_loss / len(valid_loader)
        accuracy = correct / total
        epoch_valid_losses.append(valid_loss)
        epoch_accuracies.append(accuracy)

        print(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f} | Acc: {accuracy*100:.2f}%")
        
        # NEW: Tell the scheduler what the validation loss was so it can adjust!
        scheduler.step(valid_loss)
        
        if accuracy > best_acc:
            best_acc = accuracy
            save_name = f"model_{exp_name}.pth"
            torch.save(model.state_dict(), os.path.join('saved_models', save_name))

    print("\nTraining Complete!")
    
    # ---------------------------------------------------------
    # METRICS & VISUALIZATION
    # ---------------------------------------------------------
    plot_loss_curve(args.epochs, epoch_train_losses, epoch_valid_losses, epoch_accuracies, save_path=os.path.join('outputs', f"loss_{exp_name}.png"))

    class_labels = {v: k for k, v in full_dataset.label_mapping.items()}
    model.load_state_dict(torch.load(os.path.join('saved_models', save_name)))
    
    try:
        plot_confusion_matrix(valid_loader, class_labels, model, args, device, save_path=os.path.join('outputs', f"conf_mat_{exp_name}.png"))
    except Exception as e:
        print(f"[WARNING] Could not run plot_confusion_matrix automatically. Error: {e}")

if __name__ == "__main__":
    main()