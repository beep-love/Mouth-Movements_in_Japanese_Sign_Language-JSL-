import os
import argparse
import cv2
import torch
import numpy as np
import random
from collections import defaultdict
from torchvision import transforms

from src.data.jsl_dataset import MultiVideoFaceBodySequenceDataset

# ==========================================
# GRAPH EDGE DEFINITIONS (From ST-GCN)
# ==========================================
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # Pinky
    (0, 17)                                   # Palm Closure
]

LIP_EDGES = [(i, (i + 1) % 20) for i in range(20)]

POSE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7), # Nose to L Ear
    (0, 4), (4, 5), (5, 6), (6, 8), # Nose to R Ear
    (9, 10),                        # Shoulders
    (9, 11), (11, 13),              # Left Arm
    (10, 12), (12, 14)              # Right Arm
]

# Base 17-point MediaPipe cheek triangulation
CHEEK_LOCAL = [
    (0,1), (1,2), (2,3), (4,5), (5,6), (6,7), (8,9), (9,10), (10,11), (11,12), (13,14), (14,15), (15,16),
    (0,4), (0,5), (1,5), (1,6), (2,6), (3,6), (3,7),
    (4,8), (4,9), (5,9), (5,10), (6,10), (6,11), (7,11), (7,12),
    (9,13), (10,13), (10,14), (10,15), (11,15), (11,16), (12,16)
]
# Mirror for the left side (offset by 17)
CHEEK_EDGES = CHEEK_LOCAL + [(u+17, v+17) for u, v in CHEEK_LOCAL]


# ==========================================
# DRAWING UTILITIES
# ==========================================
def draw_graph_edges(img, landmarks, edges, color, thickness=1):
    """Utility to draw connecting lines between nodes based on the ST-GCN graph."""
    if landmarks is None or len(landmarks) == 0:
        return img
        
    h, w, _ = img.shape
    for u, v in edges:
        # Safety check in case a crop is missing nodes
        if u >= len(landmarks) or v >= len(landmarks):
            continue
            
        pt1 = landmarks[u]
        pt2 = landmarks[v]
        
        # Skip Forward-filled/Zero-padded missing points
        if (pt1[0].item() == 0.0 and pt1[1].item() == 0.0) or (pt2[0].item() == 0.0 and pt2[1].item() == 0.0):
            continue
            
        x1, y1 = int(pt1[0].item() * w), int(pt1[1].item() * h)
        x2, y2 = int(pt2[0].item() * w), int(pt2[1].item() * h)
        
        # Draw only if both points are within image bounds
        if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            
    return img

def draw_landmarks(img, landmarks, color, radius=2):
    """Utility to draw normalized [0, 1] PyTorch landmarks onto an OpenCV image."""
    if landmarks is None or len(landmarks) == 0:
        return img
    
    h, w, _ = img.shape
    for point in landmarks:
        if point[0].item() == 0.0 and point[1].item() == 0.0:
            continue
            
        x = int(point[0].item() * w)
        y = int(point[1].item() * h)
        
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x, y), radius, color, -1)
    return img

def tensor_to_cv2(tensor_img):
    img_np = tensor_img.permute(1, 2, 0).numpy() * 255.0
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

def draw_text_with_bg(img, text, position, font_scale=0.5, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(img, (x - 2, y - text_size[1] - 4), (x + text_size[0] + 2, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness)

# ==========================================
# MAIN ROUTINE
# ==========================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, required=True, help="Path to your extracted_features directory")
    ap.add_argument("--save_dir", type=str, required=True, help="Where to save the MP4s")
    ap.add_argument("--fps", type=int, default=5, help="Playback speed (15 is good for presentations)")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading dataset map with Dual Transforms...")
    face_t = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    body_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = MultiVideoFaceBodySequenceDataset(
        root_dir=args.root_dir, 
        task='classification', 
        mode='both',
        face_transform=face_t,
        body_transform=body_t,
        return_landmarks=True,
        downsample=1,
        min_seq_len=30,
        max_seq_len=90 
    )

    target_per_class = 10
    saved_counts = defaultdict(int)
    idx_to_label = {v: k for k, v in dataset.label_mapping.items()}
    
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    print(f"Dataset loaded. Searching for {target_per_class} examples per class...")

    for idx in indices:
        sample = dataset[idx]
        label_str = idx_to_label[sample['label'].item()]
        
        if saved_counts[label_str] >= target_per_class:
            if all(saved_counts[lbl] >= target_per_class for lbl in dataset.label_mapping.keys()):
                print("\n Successfully generated all presentation videos!")
                break
            continue

        seq_len = sample['length']
        video_w = 448 
        video_h = 224
        
        vid_num = saved_counts[label_str] + 1
        safe_label = label_str.replace(' ', '_')
        out_path = os.path.join(args.save_dir, f"{safe_label}_{vid_num}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, args.fps, (video_w, video_h))

        for f_idx in range(seq_len):
            face_img_raw = tensor_to_cv2(sample['face_seq'][f_idx])
            body_img = tensor_to_cv2(sample['body_seq'][f_idx])

            # Upscale face
            face_img = cv2.resize(face_img_raw, (224, 224), interpolation=cv2.INTER_NEAREST)

            # --- DRAW FACE GRAPH ---
            # Edges
            face_img = draw_graph_edges(face_img, sample['lm_face_outer_lips'][f_idx], LIP_EDGES, (0, 200, 200))
            face_img = draw_graph_edges(face_img, sample['lm_face_inner_lips'][f_idx], LIP_EDGES, (0, 200, 200))
            if 'lm_face_cheeks' in sample and sample['lm_face_cheeks'].shape[1] > 0:
                face_img = draw_graph_edges(face_img, sample['lm_face_cheeks'][f_idx], CHEEK_EDGES, (0, 200, 200))
            # Nodes
            face_img = draw_landmarks(face_img, sample['lm_face_outer_lips'][f_idx], (0, 255, 255))
            face_img = draw_landmarks(face_img, sample['lm_face_inner_lips'][f_idx], (0, 255, 255))
            if 'lm_face_cheeks' in sample and sample['lm_face_cheeks'].shape[1] > 0:
                face_img = draw_landmarks(face_img, sample['lm_face_cheeks'][f_idx], (0, 255, 255))

            # --- DRAW BODY GRAPH ---
            # Edges
            body_img = draw_graph_edges(body_img, sample['lm_body_outer_lips'][f_idx], LIP_EDGES, (0, 200, 200))
            body_img = draw_graph_edges(body_img, sample['lm_body_inner_lips'][f_idx], LIP_EDGES, (0, 200, 200))
            body_img = draw_graph_edges(body_img, sample['lm_pose'][f_idx], POSE_EDGES, (0, 200, 0))
            body_img = draw_graph_edges(body_img, sample['lm_l_hand'][f_idx], HAND_EDGES, (0, 0, 200))
            body_img = draw_graph_edges(body_img, sample['lm_r_hand'][f_idx], HAND_EDGES, (200, 0, 0))
            if 'lm_body_cheeks' in sample and sample['lm_body_cheeks'].shape[1] > 0:
                body_img = draw_graph_edges(body_img, sample['lm_body_cheeks'][f_idx], CHEEK_EDGES, (0, 200, 200))
            
            # Nodes
            body_img = draw_landmarks(body_img, sample['lm_body_outer_lips'][f_idx], (0, 255, 255))
            body_img = draw_landmarks(body_img, sample['lm_body_inner_lips'][f_idx], (0, 255, 255))
            if 'lm_body_cheeks' in sample and sample['lm_body_cheeks'].shape[1] > 0:
                body_img = draw_landmarks(body_img, sample['lm_body_cheeks'][f_idx], (0, 255, 255))
            body_img = draw_landmarks(body_img, sample['lm_pose'][f_idx], (0, 255, 0), radius=3)
            body_img = draw_landmarks(body_img, sample['lm_l_hand'][f_idx], (0, 0, 255))
            body_img = draw_landmarks(body_img, sample['lm_r_hand'][f_idx], (255, 0, 0))

            combined_frame = np.hstack((face_img, body_img))

            draw_text_with_bg(combined_frame, f"Class: {label_str}", (10, 20))
            draw_text_with_bg(combined_frame, f"Frame: {f_idx+1}/{seq_len}", (10, 45))
            draw_text_with_bg(combined_frame, "Face Crop (64x64 upscaled)", (10, video_h - 5))
            draw_text_with_bg(combined_frame, "Body Crop (224px)", (224 + 10, video_h - 5))

            out.write(combined_frame)

        out.release()
        saved_counts[label_str] += 1
        print(f"Saved {out_path} ({seq_len} frames)")

if __name__ == "__main__":
    main()