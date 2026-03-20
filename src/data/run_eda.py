import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from collections import defaultdict

from jsl_dataset import MultiVideoFaceBodySequenceDataset

def analyze_sequences(root_dir, task, save_dir, downsample, min_seq_len, max_seq_len):
    """Analyzes class distributions, sequence lengths, and missing hands."""
    print(f"\n--- Running Sequence Analysis for Task: {task.upper()} ---")
    task_dir = os.path.join(save_dir, task)
    os.makedirs(task_dir, exist_ok=True)

    dataset = MultiVideoFaceBodySequenceDataset(
        root_dir=root_dir, task=task, mode='no_img', 
        return_landmarks=True, min_seq_len=min_seq_len, max_seq_len=max_seq_len, downsample=downsample
    )

    if len(dataset) == 0:
        print(f"[WARN] No samples found for task '{task}'. Skipping...")
        return

    class_counts = defaultdict(int)
    class_lengths = defaultdict(list)
    all_lengths = []
    
    missing_l_hand = 0
    missing_r_hand = 0
    total_frames = 0

    for idx in tqdm(range(len(dataset)), desc=f"Scanning {task} Sequences"):
        sample = dataset[idx]
        label_str = list(dataset.label_mapping.keys())[list(dataset.label_mapping.values()).index(sample['label'].item())]
        length = sample['length']
        
        class_counts[label_str] += 1
        class_lengths[label_str].append(length)
        all_lengths.append(length)

        # Detect Forward-Filled (Frozen) Hands + Pre-appearance Zeros
        l_hand = sample['lm_l_hand'] 
        r_hand = sample['lm_r_hand']

        l_hand_zeros = (l_hand.abs().sum(dim=(1, 2)) == 0).sum().item()
        r_hand_zeros = (r_hand.abs().sum(dim=(1, 2)) == 0).sum().item()

        l_hand_frozen, r_hand_frozen = 0, 0
        if length > 1:
            l_diff = (l_hand[1:] - l_hand[:-1]).abs().sum(dim=(1, 2))
            r_diff = (r_hand[1:] - r_hand[:-1]).abs().sum(dim=(1, 2))
            l_hand_frozen = ((l_diff == 0) & (l_hand[1:].abs().sum(dim=(1, 2)) != 0)).sum().item()
            r_hand_frozen = ((r_diff == 0) & (r_hand[1:].abs().sum(dim=(1, 2)) != 0)).sum().item()

        missing_l_hand += (l_hand_zeros + l_hand_frozen)
        missing_r_hand += (r_hand_zeros + r_hand_frozen)
        total_frames += length

    # Print Text Report
    print(f"\n[ {task.upper()} REPORT ]")
    print(f"Total Sequences: {len(all_lengths)}")
    print(f"Total Frames: {total_frames}")
    if total_frames > 0:
        print(f"Frames missing Left Hand: {missing_l_hand} ({(missing_l_hand/total_frames)*100:.1f}%)")
        print(f"Frames missing Right Hand: {missing_r_hand} ({(missing_r_hand/total_frames)*100:.1f}%)\n")
    
    for label in sorted(class_counts):
        lengths = class_lengths[label]
        print(f"{label:<20} | count: {len(lengths):<5} | avg_len: {np.mean(lengths):.1f} | min: {min(lengths):<3} | max: {max(lengths)}")

    # Plot 1: Class Distribution
    labels, counts = zip(*sorted(class_counts.items()))
    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, counts, color='skyblue')
    plt.title(f"{task.capitalize()} - Class Distribution")
    plt.xticks(rotation=15)
    
    # Add counts to the top of the bars
    for bar in bars:
        yval = bar.get_height()
        # Offset the text slightly above the bar based on the max count
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(counts) * 0.01), int(yval), ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(task_dir, "class_distribution.png"))
    plt.close()

    # Plot 2: Sequence Length Boxplot
    plt.figure(figsize=(10, 5))
    plt.boxplot([class_lengths[l] for l in labels], tick_labels=labels)
    plt.title(f"{task.capitalize()} - Sequence Lengths per Class")
    plt.ylabel("Number of Frames")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(task_dir, "length_boxplot.png"))
    plt.close()

    # Plot 3: Histogram of All Lengths
    plt.figure(figsize=(10, 5))
    bins = list(range(0, max(all_lengths) + 10, max(1, max(all_lengths)//20)))
    plt.hist(all_lengths, bins=bins, color='mediumpurple', edgecolor='black')
    plt.title(f"{task.capitalize()} - Overall Sequence Length Histogram")
    plt.xlabel("Number of Frames")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(task_dir, "length_histogram.png"))
    plt.close()


def analyze_images(root_dir, save_dir):
    """Analyzes resolutions and calculates Mean/Std for both Face and Body images."""
    print("\n--- Running Image Statistics (Mean, Std, Resolution) ---")
    transform = transforms.ToTensor()
    
    stats = {
        'face': {'sizes': [], 'sum_mean': torch.zeros(3), 'sum_std': torch.zeros(3), 'pixels': 0, 'count': 0},
        'body': {'sizes': [], 'sum_mean': torch.zeros(3), 'sum_std': torch.zeros(3), 'pixels': 0, 'count': 0}
    }

    video_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for video in tqdm(video_dirs, desc="Scanning Image Folders"):
        for crop_type in ['face', 'body']:
            img_dir = os.path.join(root_dir, video, crop_type)
            if not os.path.exists(img_dir): continue
            
            for file in os.listdir(img_dir):
                if file.endswith(".png"):
                    img_path = os.path.join(img_dir, file)
                    img = Image.open(img_path).convert("RGB")
                    
                    # Size Tracking
                    stats[crop_type]['sizes'].append(img.size)
                    
                    # Mean/Std Tracking
                    tensor = transform(img)
                    pixels = tensor.numel() // 3
                    stats[crop_type]['pixels'] += pixels
                    stats[crop_type]['sum_mean'] += tensor.sum(dim=[1, 2])
                    stats[crop_type]['sum_std'] += (tensor ** 2).sum(dim=[1, 2])
                    stats[crop_type]['count'] += 1

    # Calculate and Print Results
    for crop_type, data in stats.items():
        if data['count'] == 0: continue
        
        widths, heights = zip(*data['sizes'])
        mean = data['sum_mean'] / data['pixels']
        std = (data['sum_std'] / data['pixels'] - mean ** 2).sqrt()
        
        print(f"\n[ {crop_type.upper()} IMAGE STATS ]")
        print(f"Total Images: {data['count']}")
        print(f"Min Size: {np.min(widths)}x{np.min(heights)}")
        print(f"Max Size: {np.max(widths)}x{np.max(heights)}")
        print(f"Avg Size: {np.mean(widths):.1f}x{np.mean(heights):.1f}")
        print(f"Dataset Mean: {[round(x, 4) for x in mean.tolist()]}")
        print(f"Dataset Std:  {[round(x, 4) for x in std.tolist()]}")

        # Image Size Scatter Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(widths, heights, alpha=0.1, color='orange' if crop_type == 'face' else 'green')
        plt.title(f"{crop_type.capitalize()} - Image Resolution Distribution")
        plt.xlabel("Width (px)")
        plt.ylabel("Height (px)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{crop_type}_resolution_scatter.png"))
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, required=True, help="Path to your extracted_features directory")
    ap.add_argument("--base_save_dir", type=str, required=True, help="Where to save the plots")

    # NEW PARAMETERS FOR FILTERING
    ap.add_argument("--downsample", type=int, default=0, help="0: None, 1: Balance to 2nd largest class, 2: Balance to smallest class")
    ap.add_argument("--min_seq_len", type=int, default=None, help="Minimum allowed frames per clip")
    ap.add_argument("--max_seq_len", type=int, default=None, help="Maximum allowed frames per clip")

    args = ap.parse_args()

    # Automatically generate a smart folder name based on the parameters!
    folder_suffix = f"_ds{args.downsample}"
    if args.min_seq_len is not None:
        folder_suffix += f"_min{args.min_seq_len}"
    if args.max_seq_len is not None:
        folder_suffix += f"_max{args.max_seq_len}"
        
    final_save_dir = f"{args.base_save_dir}{folder_suffix}"
    os.makedirs(final_save_dir, exist_ok=True)
    
    print(f"\n[INFO] Saving all EDA outputs to: {final_save_dir}")

    # Pass the parameters into the sequence analyzer
    analyze_sequences(args.root_dir, task='classification', save_dir=final_save_dir, 
                      downsample=args.downsample, min_seq_len=args.min_seq_len, max_seq_len=args.max_seq_len)
    
    analyze_sequences(args.root_dir, task='localization', save_dir=final_save_dir, 
                      downsample=args.downsample, min_seq_len=args.min_seq_len, max_seq_len=args.max_seq_len)
    
    # Deep Image Pixel Analysis (Takes a bit longer, reads the PNGs)
    analyze_images(args.root_dir, save_dir=final_save_dir)

    print(f"\n All EDA completed! Plots saved to: {os.path.abspath(final_save_dir)}")

if __name__ == "__main__":
    main()