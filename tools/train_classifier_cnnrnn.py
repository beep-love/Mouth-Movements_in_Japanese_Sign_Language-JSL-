import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np

# Imports from our modular structure
from src.data.jsl_dataset import MultiVideoFaceBodySequenceDataset, BucketedBatchSampler, collate_fn_face_body_sequence
from src.model.classification.cnn_rnn import MultimodalSequenceClassifier
from src.utils.metrics import plot_loss_curve, plot_confusion_matrix

class SubsetWithSamples(Subset):
    """Wrapper to ensure the batch sampler can still access the 'samples' list for length grouping."""
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.samples = [dataset.samples[i] for i in indices]

def main():
    parser = argparse.ArgumentParser(description="Train Multimodal Classification Model")
    parser.add_argument("--data_dir", type=str, default="./data/processed_annotations", help="Path to the processed dataset directory")
    parser.add_argument("--mode", type=str, default="both", choices=["face", "body", "both", "no_img"])
    parser.add_argument("--face_backbone", type=str, default="custom", choices=["resnet18", "custom"])
    parser.add_argument("--body_backbone", type=str, default="resnet18", choices=["resnet18", "custom"])
    parser.add_argument("--rnn_type", type=str, default="gru", choices=["lstm", "gru"])

    
    # Dynamic Landmark List for Ablation Studies
    parser.add_argument(
        "--landmarks", 
        nargs='*', 
        type=str, 
        default=[], 
        help="List of landmarks to include (e.g. --landmarks lm_face_inner_lips lm_face_outer_lips lm_pose)"
    )
    
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # DUAL TRANSFORMS
    # ---------------------------------------------------------
    # Face gets 64x64 (Optimal for Custom CNN)
    face_t = transforms.Compose([
        transforms.Resize((64, 64)),  # Custom CNN prefers smaller input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5405, 0.4073, 0.3384], std=[0.2023, 0.1721, 0.1558]) 
    ])
    
    # Body gets 224x224 (Optimal for ResNet18)
    body_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    # ---------------------------------------------------------
    # DYNAMIC LANDMARK CONFIGURATION
    # ---------------------------------------------------------
    # Lookup table for the dimensions of each landmark array (Nodes * 2 coordinates) # Total keypoints = 187 * 2 = 374 :- full face version - 40+40+68=148, full body version - 40+40+68+30+42+42=262
    LANDMARK_DIMS = {
        'lm_face_outer_lips': 20 * 2,   # 40
        'lm_face_inner_lips': 20 * 2,   # 40
        'lm_face_cheeks': 17 * 2 * 2,   # 68 keypoints for cheeks (17 per side)
        'lm_body_outer_lips': 20 * 2,   # 40
        'lm_body_inner_lips': 20 * 2,   # 40
        'lm_body_cheeks': 17 * 2 * 2,   # 68 
        'lm_pose': 15 * 2,              # 30
        'lm_l_hand': 21 * 2,            # 42
        'lm_r_hand': 21 * 2             # 42
    }

    use_landmarks = len(args.landmarks) > 0
    if use_landmarks:
        # Validate requested landmarks and calculate the exact MLP input dimension
        for lm in args.landmarks:
            if lm not in LANDMARK_DIMS:
                raise ValueError(f"Unknown landmark key '{lm}'. Available keys: {list(LANDMARK_DIMS.keys())}")
        lm_str = "-".join([lm.replace('lm_', '') for lm in args.landmarks])  # e.g., "face_inner_lips-body_outer_lips-pose"
        
        LM_DIM = sum([LANDMARK_DIMS[lm] for lm in args.landmarks])
        print(f"[INFO] Active Landmarks: {args.landmarks}")
        print(f"[INFO] Landmark MLP Input Dimension: {LM_DIM}")
    else:
        LM_DIM = None
        lm_str = "no_lm"
        print("[INFO] No landmarks requested. Operating in Image-Only mode.")

    exp_name = f"I-{args.mode}_F-{args.face_backbone}_B-{args.body_backbone}_R-{args.rnn_type}_{lm_str}"
    print(f"\n[INFO] Starting Experiment: {exp_name}\n")

    # ---------------------------------------------------------
    # DATASET & DATALOADERS
    # ---------------------------------------------------------
    full_dataset = MultiVideoFaceBodySequenceDataset(
        root_dir=args.data_dir,
        task='classification',
        mode=args.mode,
        return_landmarks=use_landmarks,
        face_transform=face_t,
        body_transform=body_t,
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
    model = MultimodalSequenceClassifier(
        mode=args.mode,
        use_landmarks=use_landmarks,
        lm_input_dim=LM_DIM,
        face_backbone=args.face_backbone,
        body_backbone=args.body_backbone,
        rnn_type=args.rnn_type,
        num_classes=4
    ).to(device)

    # criterion = nn.CrossEntropyLoss()
    # ---------------------------------------------------------
    # CALCULATE DYNAMIC CLASS WEIGHTS
    # ---------------------------------------------------------
    class_counts = torch.zeros(4)           # Assuming 4 classes
    for sample in train_dataset.samples:
        lbl = sample['label']
        
        # 1. Translate string labels to integers using dataset's mapping!
        if isinstance(lbl, str):
            lbl = full_dataset.label_mapping[lbl]
        elif torch.is_tensor(lbl):
            lbl = lbl.item()
        elif isinstance(lbl, (list, tuple, np.ndarray)):
            lbl = lbl[0] 
            
        # Ensure it is a standard Python int
        class_counts[int(lbl)] += 1
        
    print(f"\n[INFO] Train Set Class Counts: {class_counts.tolist()}")

    # Calculate inverse frequency weights
    weights = 1.0 / torch.sqrt(class_counts + 1e-6)

    # Normalize so the majority class gets a weight of ~1.0
    weights = weights / weights.min()

    # Clamp extreme weights to a reasonable range (e.g., max weight of 10) to prevent instability
    # weights = torch.clamp(weights, max=10.0)
    
    print(f"[INFO] Applied Class Weights: {weights.tolist()}\n")

    # Inject the weights into the Loss Function
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    epoch_train_losses, epoch_valid_losses, epoch_accuracies = [], [], []
    best_acc = 0.0
    os.makedirs('saved_models', exist_ok=True)

    # ---------------------------------------------------------
    # BATCH HELPER (Handles dynamic inputs)
    # ---------------------------------------------------------
    def prepare_batch(batch):
        f_seq = batch['face_seq'].to(device) if 'face_seq' in batch else None
        b_seq = batch['body_seq'].to(device) if 'body_seq' in batch else None
        lengths = batch['lengths'].to(device)
        labels = batch['labels'].to(device)
        
        l_seq = None
        if use_landmarks:
            lm_parts = []
            for key in args.landmarks: 
                if key in batch:
                    part = batch[key].to(device)
                    # Flatten [B, T, Nodes, 2] -> [B, T, Nodes*2]
                    lm_parts.append(part.view(part.size(0), part.size(1), -1))
                else:
                    raise KeyError(f"Requested landmark '{key}' was not found in the batch! Check your dataset keys.")
            
            # Concatenate the chosen arrays along the feature dimension
            l_seq = torch.cat(lm_parts, dim=-1) # Final shape: [B, T, LM_DIM]
            
        return f_seq, b_seq, l_seq, lengths, labels

    # ---------------------------------------------------------
    # TRAINING LOOP
    # ---------------------------------------------------------
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            f_seq, b_seq, l_seq, lengths, labels = prepare_batch(batch)

            optimizer.zero_grad()
            outputs = model(lengths=lengths, face_seq=f_seq, body_seq=b_seq, lm_seq=l_seq)
            
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
                f_seq, b_seq, l_seq, lengths, labels = prepare_batch(batch)
                
                outputs = model(lengths=lengths, face_seq=f_seq, body_seq=b_seq, lm_seq=l_seq)
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
        
        # Save Best Model
        if accuracy > best_acc:
            best_acc = accuracy
            # Create a smart filename so ablations don't overwrite each other
            save_name = f"model_{exp_name}.pth"
            torch.save(model.state_dict(), os.path.join('saved_models', save_name))

    print("\nTraining Complete!")
    
    # ---------------------------------------------------------
    # METRICS & VISUALIZATION
    # ---------------------------------------------------------
    args.use_landmarks = use_landmarks
    
    plot_loss_curve(args.epochs, epoch_train_losses, epoch_valid_losses, epoch_accuracies, save_path=os.path.join('outputs', f"loss_{exp_name}.png"))

    # Reverse label mapping for the confusion matrix (e.g., {0: 'Mouthing', 1: 'Others'...})
    class_labels = {v: k for k, v in full_dataset.label_mapping.items()}
    
    # Reload the best weights for the final confusion matrix
    model.load_state_dict(torch.load(os.path.join('saved_models', save_name)))
    plot_confusion_matrix(valid_loader, class_labels, model, args, device, save_path=os.path.join('outputs', f"conf_mat_{exp_name}.png"))

if __name__ == "__main__":
    main()