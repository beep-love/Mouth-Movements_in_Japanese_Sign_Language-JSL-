import matplotlib.pyplot as plt
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def plot_confusion_matrix(test_dataloader, class_labels, model, args, device, save_path="confusion_matrix.png"):
    all_labels = []
    all_predictions = []
    correct = 0
    total = 0

    # Safely get flags (Handles the differences between CNN-RNN args and ST-GCN args)
    mode = getattr(args, 'mode', 'STGCN (Kinematics)')
    
    # Inline helper to parse batches
    def prepare_batch(batch):
        f_seq = batch['face_seq'].to(device) if 'face_seq' in batch else None
        b_seq = batch['body_seq'].to(device) if 'body_seq' in batch else None
        lengths = batch['lengths'].to(device)
        labels = batch['labels'].to(device)
        
        l_seq = None
        # Safely check for landmarks (ST-GCN always uses them)
        if hasattr(args, 'landmarks') and args.landmarks:
            lm_parts = []
            for key in args.landmarks:
                if key in batch:
                    part = batch[key].to(device)
                    lm_parts.append(part.view(part.size(0), part.size(1), -1))
                else:
                    raise KeyError(f"Requested landmark '{key}' was not found in the batch! Check dataset keys.")
            l_seq = torch.cat(lm_parts, dim=-1)
            
        return f_seq, b_seq, l_seq, lengths, labels

    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            f_seq, b_seq, l_seq, lengths, labels = prepare_batch(batch)
            
            # --- DYNAMIC INFERENCE ROUTING ---
            # Check which model is currently running based on its class name
            if "STGCN" in model.__class__.__name__:
                outputs = model(l_seq, lengths)
            else:
                outputs = model(lengths=lengths, face_seq=f_seq, body_seq=b_seq, lm_seq=l_seq)
            # ------------------------------------------
            
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = correct / total
    print(f"\nFinal Test Accuracy: {accuracy*100:.2f}%")
    
    true_labels = [class_labels[label] for label in all_labels]
    predicted_labels = [class_labels[pred] for pred in all_predictions]

    conf_mat = confusion_matrix(true_labels, predicted_labels, labels=list(class_labels.values()))

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels.values(), yticklabels=class_labels.values())
    plt.title(f'Confusion Matrix (Mode: {mode})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")

def plot_loss_curve(num_epochs, epoch_train_losses, epoch_test_losses, epoch_accuracies, save_path="loss_curve.png"):
    print("Preparing metrics visualization...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), epoch_train_losses, label='Train Loss', marker='o', color='b')
    plt.plot(range(1, num_epochs + 1), epoch_test_losses, label='Val Loss', marker='o', color='r')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), epoch_accuracies, label='Accuracy', marker='o', color='g')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()