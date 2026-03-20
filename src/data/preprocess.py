import os
import argparse
import cv2
import json
import pympi
import re
import numpy as np
import mediapipe as mp
import subprocess
from tqdm import tqdm
import uuid
from typing import List, Dict

# --- CONFIGURATION ---
CLIP_DURATION = 0.7       # For Classification
GAP_THRESHOLD = 0.5
SKIP_LARGE_GAP = 0.9

WINDOW_SIZE = 0.3         # For Localization
STRIDE = 0.1              # For Localization sliding window
# ---------------------

def extract_ground_truth_segments(eaf_path: str, task: str) -> List[Dict]:
    """
    Reads the new '.with_GT.eaf' files.
    task: "classification" (reads AGREEMENT tier) or "localization" (reads LOCALIZATION tier)
    """
    eaf = pympi.Elan.Eaf(eaf_path)
    annotations = []
    
    # Find the target tiers based on the task
    target_tiers = []
    for tier_id in eaf.tiers.keys():
        if task == "classification" and tier_id.startswith("AGREEMENT_") and tier_id.endswith("_GT"):
            target_tiers.append(tier_id)
        elif task == "localization" and tier_id.startswith("LOCALIZATION_") and tier_id.endswith("_GT"):
            target_tiers.append(tier_id)
            
    # Extract and format the data
    for tier_id in target_tiers:
        pid = tier_id.split('_')[2] # Extract Participant ID (e.g., AGREEMENT_IS_03_20M_GT -> 03)
        
        items = eaf.tiers[tier_id][0].items()
        sorted_anns = sorted(items, key=lambda x: eaf.timeslots[x[1][0]])
        
        total_video_duration = max(eaf.timeslots.values()) / 1000.0

        if task == "classification":
            last_end_time = 0.0
            
            for ann_id, (start, end, label, _) in sorted_anns:
                start_time = eaf.timeslots[start] / 1000.0
                end_time = eaf.timeslots[end] / 1000.0
                gap = start_time - last_end_time

                # Moderate gap: insert one no-mouth-movement label
                if GAP_THRESHOLD < gap <= SKIP_LARGE_GAP:
                    annotations.append({
                        'participant_id': pid,
                        'start_time': last_end_time,
                        'end_time': start_time,
                        'label': 'no-mouth-movement',
                        'task': 'classification'
                    })

                # Large gap: divide into CLIP_DURATION segments
                elif gap > SKIP_LARGE_GAP:
                    gap_start = last_end_time
                    while gap_start + CLIP_DURATION <= start_time:
                        annotations.append({
                            'participant_id': pid,
                            'start_time': gap_start,
                            'end_time': gap_start + CLIP_DURATION,
                            'label': 'no-mouth-movement',
                            'task': 'classification'
                        })
                        gap_start += CLIP_DURATION

                # Add current active annotation
                annotations.append({
                    'participant_id': pid,
                    'start_time': start_time,
                    'end_time': end_time,
                    'label': label,
                    'task': 'classification'
                })
                last_end_time = end_time

            # Final tail after last annotation
            if last_end_time < total_video_duration:
                while last_end_time + CLIP_DURATION <= total_video_duration:
                    annotations.append({
                        'participant_id': pid,
                        'start_time': last_end_time,
                        'end_time': last_end_time + CLIP_DURATION,
                        'label': 'no-mouth-movement',
                        'task': 'classification'
                    })
                    last_end_time += CLIP_DURATION
                
        elif task == "localization":
            # For localization, we generate sliding windows (e.g., 0.3s window, 0.1s stride)
            # over the ENTIRE video. If the center of the window hits an active block, label it as "movement", otherwise "no-mouth-movement".
            active_ranges = [(eaf.timeslots[s]/1000.0, eaf.timeslots[e]/1000.0) for _, (s, e, _, _) in sorted_anns]
            
            current_time = 0.0
            while current_time + WINDOW_SIZE <= total_video_duration:
                win_start = current_time
                win_end = current_time + WINDOW_SIZE
                win_center = (win_start + win_end) / 2.0
                
                # Check if window center falls inside any "MouthActive" GT block
                is_active = any(start <= win_center <= end for start, end in active_ranges)
                
                annotations.append({
                    'participant_id': pid,
                    'start_time': win_start,
                    'end_time': win_end,
                    'label': "movement" if is_active else "no-mouth-movement",
                    'task': 'localization'
                })
                current_time += STRIDE

    return annotations

def extract_videos_and_landmarks(video_annotation_list, save_root, extract_images=True, padding_ratio=0.05):
    """Unified MediaPipe extraction (Linear One-Pass + Smart Image Caching)."""
    os.makedirs(save_root, exist_ok=True)
    
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False, 
        model_complexity=1, 
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
    MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
    CHEEK = [
        # === RIGHT CHEEK (Image Left) ===
        117, 118, 119, 120,  # Row 1: Top, under eye
        123, 50,  101, 100,  # Row 2: Upper cheek
        147, 187, 205, 36,  142,  # Row 3: Mid cheek, max puff region (5 nodes)
        207, 216, 206, 203,  # Row 4: Bottom, near jaw/mouth corner

        # === LEFT CHEEK (Image Right - perfectly mirrored) ===
        346, 347, 348, 349,  # Row 1: Top, under eye
        352, 280, 330, 329,  # Row 2: Upper cheek
        376, 411, 425, 266, 371,  # Row 3: Mid cheek, max puff region (5 nodes)
        427, 436, 426, 423   # Row 4: Bottom, near jaw/mouth corner
    ]

    def crop_with_padding(frame, points, pad_ratio):
        if len(points) == 0:
            return None, (0, 0, 1, 1)
        h, w, _ = frame.shape
        points_arr = np.array(points)
        xs, ys = points_arr[:, 0], points_arr[:, 1]
        min_x, max_x = np.min(xs), np.max(xs)
        min_y, max_y = np.min(ys), np.max(ys)
        
        pad_x = pad_ratio * (max_x - min_x)
        pad_y = pad_ratio * (max_y - min_y)
        
        x1, y1 = int(max(min_x - pad_x, 0)), int(max(min_y - pad_y, 0))
        x2, y2 = int(min(max_x + pad_x, w)), int(min(max_y + pad_y, h))
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    for video_path, annotations in tqdm(video_annotation_list):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_save_dir = os.path.join(save_root, video_name)
        json_path = os.path.join(video_save_dir, f"{video_name}.json")

        if os.path.exists(json_path):
            print(f"Skipping {video_name} as it has already been processed.")
            continue

        os.makedirs(video_save_dir, exist_ok=True)
        if extract_images:
            os.makedirs(os.path.join(video_save_dir, "face"), exist_ok=True)
            os.makedirs(os.path.join(video_save_dir, "body"), exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)

        video_metadata = {}
        part_ids = sorted(set(ann['participant_id'] for ann in annotations))

        # --- ONE PASS PER PARTICIPANT ---
        for index, pid in enumerate(part_ids):
            # Gather all annotations for this specific person
            pid_anns = []
            for ann in annotations:
                if ann['participant_id'] == pid:
                    # Give each segment a unique ID so dict keys don't overwrite
                    ann['seg_key'] = f"{ann['task']}_{pid}_{ann['start_time']:.2f}_{ann['end_time']:.2f}_{ann['label']}_{str(uuid.uuid4())[:8]}"
                    ann['start_f'] = int(ann['start_time'] * frame_rate)
                    ann['end_f'] = int(ann['end_time'] * frame_rate)
                    pid_anns.append(ann)
                    video_metadata[ann['seg_key']] = [] # Initialize empty list for this segment

            if not pid_anns:
                continue

            # Find the absolute start and end frames for this person
            min_frame = min(a['start_f'] for a in pid_anns)
            max_frame = max(a['end_f'] for a in pid_anns)

            # Jump to the very first frame they appear in, then read linearly
            cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame)
            
            last_pose = [[0.0, 0.0]] * 15
            last_left_hand = [[0.0, 0.0]] * 21
            last_right_hand = [[0.0, 0.0]] * 21

            for frame_idx in range(min_frame, max_frame):
                ret, frame = cap.read()
                if not ret: break

                # Find which segments need this current frame
                active_anns = [a for a in pid_anns if a['start_f'] <= frame_idx < a['end_f']]
                if not active_anns:
                    continue # Skip MediaPipe entirely if this frame is in a gap between segments!

                half = frame[:, frame.shape[1]//2:] if index == 1 else frame[:, :frame.shape[1]//2]
                front = half[:half.shape[0] // 2, :]

                rgb = cv2.cvtColor(front, cv2.COLOR_BGR2RGB)
                res = holistic.process(rgb)
                
                if not res.face_landmarks:
                    continue # Drop frame if no face, dataset padding will handle it
                    
                h_img, w_img, _ = front.shape

                # Face Math
                face_coords = np.array([[lm.x * w_img, lm.y * h_img] for lm in res.face_landmarks.landmark])
                m_out, m_in, cheek = face_coords[MOUTH_OUTER], face_coords[MOUTH_INNER], face_coords[CHEEK]

                # Body Math + Forward Fill
                curr_visible_points = list(face_coords) 
                if res.pose_landmarks:
                    last_pose = [[lm.x * w_img, lm.y * h_img] for i, lm in enumerate(res.pose_landmarks.landmark) if i < 17 and i not in [9, 10]] # doesnot include mouth points
                    curr_visible_points.extend(last_pose)
                if res.left_hand_landmarks:
                    last_left_hand = [[lm.x * w_img, lm.y * h_img] for lm in res.left_hand_landmarks.landmark]
                    curr_visible_points.extend(last_left_hand)
                if res.right_hand_landmarks:
                    last_right_hand = [[lm.x * w_img, lm.y * h_img] for lm in res.right_hand_landmarks.landmark]
                    curr_visible_points.extend(last_right_hand)

                # Crops
                face_crop, (fx1, fy1, fx2, fy2) = crop_with_padding(front, face_coords, pad_ratio=-0.05) # Negative padding to ensure we don't include too much non-face area
                fw, fh = max(fx2 - fx1, 1), max(fy2 - fy1, 1)

                body_crop, (bx1, by1, bx2, by2) = crop_with_padding(front, curr_visible_points, pad_ratio=padding_ratio)
                bw, bh = max(bx2 - bx1, 1), max(by2 - by1, 1)

                # Smart Image Saving (Only save if requested AND part of a classification task)
                f_path, b_path = None, None
                needs_image = extract_images and any(a['task'] == 'classification' for a in active_anns)

                if needs_image:
                    # Name the file by its absolute frame number so it can be safely shared
                    f_path = f"face/{video_name}_{pid}_f{frame_idx:06d}.png"
                    b_path = f"body/{video_name}_{pid}_f{frame_idx:06d}.png"
                    
                    cv2.imwrite(os.path.join(video_save_dir, f_path), face_crop)
                    if body_crop is not None and body_crop.size > 0:
                        cv2.imwrite(os.path.join(video_save_dir, b_path), body_crop)
                    else:
                        b_path = None

                # Build JSON Entry
                entry = {
                    'face_path': f_path,
                    'body_path': b_path,
                    'mouth_outer_norm_face': ((m_out - [fx1, fy1]) / [fw, fh]).tolist(),
                    'mouth_inner_norm_face': ((m_in - [fx1, fy1]) / [fw, fh]).tolist(),
                    'cheek_norm_face': ((cheek - [fx1, fy1]) / [fw, fh]).tolist(),
                    'mouth_outer_norm_body': ((m_out - [bx1, by1]) / [bw, bh]).tolist(),
                    'mouth_inner_norm_body': ((m_in - [bx1, by1]) / [bw, bh]).tolist(),
                    'cheek_norm_body': ((cheek - [bx1, by1]) / [bw, bh]).tolist(),
                    'pose_norm_body': ((np.array(last_pose) - [bx1, by1]) / [bw, bh]).tolist(),
                    'left_hand_norm_body': ((np.array(last_left_hand) - [bx1, by1]) / [bw, bh]).tolist(),
                    'right_hand_norm_body': ((np.array(last_right_hand) - [bx1, by1]) / [bw, bh]).tolist()
                }
                
                # Append this math to ALL segments that require this frame!
                for ann in active_anns:
                    video_metadata[ann['seg_key']].append(entry)

        # Save JSON when both participants are fully processed
        with open(json_path, 'w') as f:
            json.dump(video_metadata, f, indent=2)
            
        cap.release()
    holistic.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_folder", required=True, help="Path to folder containing raw videos")
    ap.add_argument("--eaf_folder", required=True, help="Path to folder containing .with_GT.eaf files")
    ap.add_argument("--save_root", required=True, help="Root directory to save processed JSONs and images")
    ap.add_argument("--extract_images", action='store_true', help="Whether to extract and save face/body crops as images")

    args = ap.parse_args()
    video_folder = args.video_folder
    eaf_folder = args.eaf_folder
    save_root = args.save_root
    extract_images = args.extract_images

    # Extract Data
    video_annotation_list = []
    for fname in os.listdir(video_folder):
        if fname.endswith(".mp4"):
            base = os.path.splitext(fname)[0]
            video_path = os.path.join(video_folder, fname)
            
            # Look for the Super-EAF file
            eaf_path = os.path.join(eaf_folder, base + ".with_GT.eaf")
            
            if os.path.exists(eaf_path):
                # Run Classification Parsing (Intersection blocks)
                cls_anns = extract_ground_truth_segments(eaf_path, task="classification")
                print(f"Extracted {len(cls_anns)} classification segments from {eaf_path}")
                
                # Run Localization Parsing (Sliding window over Union blocks)
                loc_anns = extract_ground_truth_segments(eaf_path, task="localization")
                print(f"Extracted {len(loc_anns)} localization segments from {eaf_path}")
                
                # Combine them
                video_annotation_list.append((video_path, cls_anns + loc_anns))

    print(f"Starting extraction for {len(video_annotation_list)} videos...")
    # NOTE: Set extract_images=False if we only want landmarks and skip saving images
    extract_videos_and_landmarks(video_annotation_list, save_root, extract_images=extract_images)

if __name__ == "__main__":
    main()