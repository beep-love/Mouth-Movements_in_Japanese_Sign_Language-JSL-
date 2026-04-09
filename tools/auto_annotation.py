import os
import argparse
import glob
import cv2
import torch
import numpy as np
import mediapipe as mp
import pympi
import math

from src.model.classification.st_gcn import STGCN_Classifier

# --- CONSTANTS ---
WINDOW_SIZE = 0.3
STRIDE = 0.1

MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
CHEEK = [
    117, 118, 119, 120, 123, 50, 101, 100, 147, 187, 205, 36, 142, 207, 216, 206, 203,
    346, 347, 348, 349, 352, 280, 330, 329, 376, 411, 425, 266, 371, 427, 436, 426, 423
]

LANDMARK_DIMS = {
    'lm_face_outer_lips': 40, 'lm_face_inner_lips': 40, 'lm_face_cheeks': 68,
    'lm_body_outer_lips': 40, 'lm_body_inner_lips': 40, 'lm_body_cheeks': 68,
    'lm_pose': 30, 'lm_l_hand': 42, 'lm_r_hand': 42
}

# The exact mapping from jsl_dataset.py
REVERSE_CLS_MAPPING = {
    0: 'no-mouth-movement', 
    1: 'Mouthing', 
    2: 'MouthGesture', 
    3: 'Others'
}

def crop_with_padding(points, pad_ratio, w, h):
    """Calculates bounding box for normalization."""
    if len(points) == 0: return 0, 0, 1, 1
    points_arr = np.array(points)
    xs, ys = points_arr[:, 0], points_arr[:, 1]
    min_x, max_x = np.min(xs), np.max(xs)
    min_y, max_y = np.min(ys), np.max(ys)
    
    pad_x = pad_ratio * (max_x - min_x)
    pad_y = pad_ratio * (max_y - min_y)
    
    x1, y1 = max(min_x - pad_x, 0), max(min_y - pad_y, 0)
    x2, y2 = min(max_x + pad_x, w), min(max_y + pad_y, h)
    return x1, y1, max(x2 - x1, 1), max(y2 - y1, 1)

def main():
    parser = argparse.ArgumentParser(description="End-to-End ELAN Auto-Annotator")
    parser.add_argument("--video_dir", type=str, required=True, help="Folder with raw .mp4 videos")
    parser.add_argument("--eaf_dir", type=str, required=True, help="Folder with existing .eaf files")
    parser.add_argument("--out_dir", type=str, required=True, help="Folder to save updated .eaf files")
    parser.add_argument("--loc_weights", type=str, required=True, help="Path to binary localization .pth")
    parser.add_argument("--cls_weights", type=str, required=True, help="Path to 4-class classification .pth")
    parser.add_argument("--landmarks", nargs='*', type=str, required=True, help="Landmarks (must match training)")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Localization confidence threshold")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initializing Dual-STGCN Pipeline on {device}...")

    # 1. LOAD BOTH MODELS
    model_loc = STGCN_Classifier(
        active_landmarks=args.landmarks, landmark_dims=LANDMARK_DIMS, num_classes=2, in_channels=2
    ).to(device)
    model_loc.load_state_dict(torch.load(args.loc_weights, map_location=device))
    model_loc.eval()

    model_cls = STGCN_Classifier(
        active_landmarks=args.landmarks, landmark_dims=LANDMARK_DIMS, num_classes=4, in_channels=2
    ).to(device)
    model_cls.load_state_dict(torch.load(args.cls_weights, map_location=device))
    model_cls.eval()

    # 2. BATCH PROCESSING LOOP
    video_files = glob.glob(os.path.join(args.video_dir, "*.mp4"))
    
    for video_path in video_files:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        eaf_path = os.path.join(args.eaf_dir, base_name + ".eaf")
        
        if not os.path.exists(eaf_path):
            # Try replacing underscores with hyphens just in case naming conventions mismatch
            eaf_path_alt = os.path.join(args.eaf_dir, base_name.replace('_', '-') + ".eaf")
            if os.path.exists(eaf_path_alt):
                eaf_path = eaf_path_alt
            else:
                print(f"[WARN] No matching .eaf found for {base_name}. Skipping.")
                continue

        print(f"\n========================================")
        print(f"Processing: {base_name}")
        print(f"========================================")

        # 3. KINEMATIC EXTRACTION
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        window_frames = int(WINDOW_SIZE * fps)
        stride_frames = int(STRIDE * fps)

        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(static_image_mode=False, refine_face_landmarks=True)

        extracted_data = {lm: [] for lm in args.landmarks}
        last_known = {lm: np.zeros((LANDMARK_DIMS[lm]//2, 2)) for lm in args.landmarks}

        print("[INFO] Phase 1: Extracting MediaPipe coordinates...")
        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret: break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)
            h_img, w_img, _ = frame.shape

            if res.face_landmarks:
                face_coords = np.array([[lm.x * w_img, lm.y * h_img] for lm in res.face_landmarks.landmark])
                m_out, m_in, cheek = face_coords[MOUTH_OUTER], face_coords[MOUTH_INNER], face_coords[CHEEK]
                fx1, fy1, fw, fh = crop_with_padding(face_coords, -0.05, w_img, h_img)
                
                if 'lm_face_outer_lips' in args.landmarks: last_known['lm_face_outer_lips'] = (m_out - [fx1, fy1]) / [fw, fh]
                if 'lm_face_inner_lips' in args.landmarks: last_known['lm_face_inner_lips'] = (m_in - [fx1, fy1]) / [fw, fh]
                if 'lm_face_cheeks' in args.landmarks: last_known['lm_face_cheeks'] = (cheek - [fx1, fy1]) / [fw, fh]

            # Forward-fill to protect against dropped frames
            for lm in args.landmarks:
                extracted_data[lm].append(last_known[lm])

        cap.release()
        holistic.close()

        # 4. LOCALIZATION (Sliding Window)
        print("[INFO] Phase 2: Running Binary Localization...")
        num_extracted = len(extracted_data[args.landmarks[0]])
        predictions = []

        with torch.no_grad():
            for start_f in range(0, num_extracted - window_frames + 1, stride_frames):
                end_f = start_f + window_frames
                
                lm_parts = []
                for lm in args.landmarks:
                    chunk = np.array(extracted_data[lm][start_f:end_f])
                    chunk = torch.tensor(chunk, dtype=torch.float32).view(window_frames, -1)
                    lm_parts.append(chunk)
                    
                window_tensor = torch.cat(lm_parts, dim=-1).unsqueeze(0).to(device)
                lengths = torch.tensor([window_frames]).to(device)

                logits = model_loc(window_tensor, lengths)
                move_prob = torch.softmax(logits, dim=1)[0][1].item()

                predictions.append({
                    "start": start_f / fps, "end": end_f / fps, "prob": move_prob
                })

        # Bin Voting
        max_time = predictions[-1]["end"] if predictions else 0
        num_bins = math.ceil(max_time / STRIDE)
        bin_votes = [0.0] * num_bins
        bin_counts = [0] * num_bins

        for p in predictions:
            s_bin = int(p["start"] // STRIDE)
            e_bin = int(p["end"] // STRIDE)
            for b in range(s_bin, e_bin):
                if b < len(bin_votes):
                    bin_votes[b] += p["prob"]
                    bin_counts[b] += 1

        active_bins = [b for b, count in enumerate(bin_counts) if count > 0 and (bin_votes[b]/count) >= args.conf_threshold]

        loc_segments = []
        if active_bins:
            curr_start = active_bins[0]
            prev_bin = active_bins[0]
            for b in active_bins[1:]:
                if b == prev_bin + 1: prev_bin = b
                else:
                    loc_segments.append((curr_start * STRIDE, (prev_bin + 1) * STRIDE))
                    curr_start, prev_bin = b, b
            loc_segments.append((curr_start * STRIDE, (prev_bin + 1) * STRIDE))

        # 5. CLASSIFICATION (On localized boundaries)
        print("[INFO] Phase 3: Classifying Localized Segments...")
        final_annotations = []
        
        with torch.no_grad():
            for start_sec, end_sec in loc_segments:
                start_f = int(start_sec * fps)
                end_f = int(end_sec * fps)
                
                # Safety check: sequence must be at least 2 frames to classify
                if end_f - start_f < 2: continue
                
                lm_parts = []
                for lm in args.landmarks:
                    chunk = np.array(extracted_data[lm][start_f:end_f])
                    chunk = torch.tensor(chunk, dtype=torch.float32).view(end_f - start_f, -1)
                    lm_parts.append(chunk)
                
                seq_tensor = torch.cat(lm_parts, dim=-1).unsqueeze(0).to(device)
                lengths = torch.tensor([end_f - start_f]).to(device)

                logits = model_cls(seq_tensor, lengths)
                pred_idx = torch.argmax(logits, dim=1).item()
                label_str = REVERSE_CLS_MAPPING[pred_idx]
                
                final_annotations.append((start_sec, end_sec, label_str))

        # 6. EAF INJECTION
        print("[INFO] Phase 4: Injecting Tiers into ELAN file...")
        eaf = pympi.Elan.Eaf(eaf_path)
        
        # We append a unique ID or label to prevent overwriting if run multiple times
        loc_tier = "AI_LOCALIZATION_PRED"
        cls_tier = "AI_CLASSIFICATION_PRED"
        
        if loc_tier not in eaf.tiers: eaf.add_tier(loc_tier)
        if cls_tier not in eaf.tiers: eaf.add_tier(cls_tier)

        for start_sec, end_sec, label in final_annotations:
            start_ms = int(start_sec * 1000)
            end_ms = int(end_sec * 1000)
            
            # Write to both tiers
            eaf.add_annotation(loc_tier, start_ms, end_ms, "MouthActive")
            eaf.add_annotation(cls_tier, start_ms, end_ms, label)

        out_path = os.path.join(args.out_dir, base_name + "_AI_Annotated.eaf")
        eaf.to_file(out_path)
        print(f"[SUCCESS] Injected {len(final_annotations)} annotations. Saved to {out_path}")

    print("\n[INFO] End-to-End Processing Complete.")

if __name__ == "__main__":
    main()