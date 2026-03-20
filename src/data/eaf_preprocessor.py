import argparse
import collections
import csv
import os
import re
import glob
import itertools
import subprocess
from typing import List, Tuple, Dict, Optional

import numpy as np
import pympi
import cv2

try:
    import mediapipe as mp
    _HAS_MP = True
except Exception:
    _HAS_MP = False

Segment = collections.namedtuple("Segment", "start end label")

NO_LABEL = "No Mouth Movement"

# -----------------------------
# Normalization, Typo Fixing & video extension handling
# -----------------------------
def norm_label(x: str) -> str:
    x = (x or "").strip()
    
    # Standardize empty/none
    if x == "" or x.lower() in {"none", "no mouth movement", "no-mouth-movement"}:
        return NO_LABEL
    
    # Fix specific typos requested
    if x == "Mouthing Mouthing": 
        return "Mouthing"
    if x == "MouthGesture MouthGesture":
        return "MouthGesture"
        
    return x

def convert_mov_to_mp4(folder_path):
    """Converts .mov to .mp4 using FFmpeg and deletes the original .mov"""
    if not folder_path or not os.path.exists(folder_path):
        return
        
    mov_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.mov')]
    if not mov_files:
        return

    print(f"\n[INFO] Found {len(mov_files)} .mov files. Starting conversion to .mp4...")
    for mov_file in mov_files:
        mov_path = os.path.join(folder_path, mov_file)
        mp4_file = mov_file.rsplit('.', 1)[0] + ".mp4"
        mp4_path = os.path.join(folder_path, mp4_file)

        if not os.path.exists(mp4_path):
            cmd = [
                "ffmpeg", "-i", mov_path, 
                "-vf", "scale=ceil(iw/2)*2:ceil(ih/2)*2", 
                "-vcodec", "libx264", "-acodec", "aac", 
                "-strict", "experimental", mp4_path
            ]
            try:
                # suppress the massive FFmpeg text output to keep the console clean
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f" -> Converted: {mp4_file}")
                
                # Safely remove the original .mov file
                os.remove(mov_path)
            except subprocess.CalledProcessError as e:
                print(f" -> [ERROR] Failed to convert {mov_file}: {e}")

def standardize_filenames(folder_path, extensions):
    """Scans a folder and replaces any hyphens in target files with underscores."""
    if not folder_path or not os.path.exists(folder_path):
        return
        
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(extensions) and '-' in fname:
            new_fname = fname.replace('-', '_')
            old_path = os.path.join(folder_path, fname)
            new_path = os.path.join(folder_path, new_fname)
            
            os.rename(old_path, new_path)
            print(f"[INFO] Renamed: {fname} -> {new_fname}")

# -----------------------------
# Tier Discovery
# -----------------------------
def discover_annotator_groups(eaf: pympi.Elan.Eaf) -> Dict[str, List[Tuple[str, str]]]:
    """
    Returns a map: { 'PID': [ (AnnotatorName, TierName), ... ] }
    """
    patt = re.compile(r"^(?P<pid>.+?)-MouthAction-(?P<ann>.+)$")
    pid_map = collections.defaultdict(list)

    for tier_name in eaf.tiers.keys():
        m = patt.match(tier_name)
        if m:
            pid = m.group("pid")
            ann = m.group("ann")
            pid_map[pid].append((ann, tier_name))
    return pid_map

def tier_segments(eaf: pympi.Elan.Eaf, tier: str) -> List[Segment]:
    items = eaf.tiers[tier][0].items()
    ts = eaf.timeslots
    segs = []
    for _ann_id, (s, e, val, _) in items:
        segs.append(Segment(ts[s] / 1000.0, ts[e] / 1000.0, norm_label(str(val))))
    segs.sort(key=lambda x: (x.start, x.end))
    return segs

# -----------------------------
# Multi-Annotator Processing Core
# -----------------------------
def get_combined_atomic_intervals(all_segments_list, tol):
    """Generates atomic intervals across N lists of segments."""
    cuts = set()
    cuts.add(0.0)
    for seg_list in all_segments_list:
        for s in seg_list:
            cuts.add(s.start)
            cuts.add(s.end)
    cuts = sorted(list(cuts))
    
    intervals = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a >= tol:
            intervals.append((a, b))
    return intervals

def get_label_at_interval(segs: List[Segment], a: float, b: float, tol: float) -> str:
    """Finds the label active during this interval for a single annotator."""
    for s in segs:
        if s.start <= a + tol and s.end >= b - tol:
            return s.label
    return NO_LABEL

def generate_consensus_tiers(pid_name, annotator_entries, eaf, tol):
    """
    Generates Consensus Ground Truth for N annotators.
    Returns: classification_intervals, localization_intervals
    """
    # Load all segments
    # annotator_entries is list of (AnnName, TierName)
    all_tier_data = []
    for ann_name, tier_name in annotator_entries:
        all_tier_data.append(tier_segments(eaf, tier_name))
    
    # Slice time based on ALL annotators
    intervals = get_combined_atomic_intervals(all_tier_data, tol)
    
    classification_gt = []
    localization_gt = []
    
    num_annotators = len(all_tier_data)
    # Threshold for Majority: Half + 1 (e.g., 3 -> 2, 2 -> 2)
    # If 2 annotators, majority is 2 (Unanimous). If 3, majority is 2.
    majority_thresh = (num_annotators // 2) + 1 if num_annotators > 2 else num_annotators

    for a, b in intervals:
        # Collect votes from all annotators for this slice
        votes = []
        has_activity = False
        
        for seg_list in all_tier_data:
            lbl = get_label_at_interval(seg_list, a, b, tol)
            votes.append(lbl)
            if lbl != NO_LABEL:
                has_activity = True
        
        # --- LOGIC 1: LOCALIZATION GT (Union) ---
        # If ANYONE sees movement, we mark it as active for localization training
        if has_activity:
            # Merge logic: extend last block if contiguous
            if localization_gt and abs(a - localization_gt[-1][1]) < 1e-9:
                localization_gt[-1] = (localization_gt[-1][0], b)
            else:
                localization_gt.append((a, b))

        # --- LOGIC 2: CLASSIFICATION GT (Majority/Consensus) ---
        # Count label frequencies (excluding Silence)
        active_votes = [v for v in votes if v != NO_LABEL]
        if not active_votes:
            continue # Everyone agrees it's silence
            
        counts = collections.Counter(active_votes)
        top_label, count = counts.most_common(1)[0]
        
        # Check if it meets majority threshold
        # For N=2: Need 2 votes (Intersection)
        # For N=3: Need 2 votes
        if count >= majority_thresh:
            if classification_gt and abs(a - classification_gt[-1][1]) < 1e-9 and classification_gt[-1][2] == top_label:
                classification_gt[-1] = (classification_gt[-1][0], b, top_label)
            else:
                classification_gt.append((a, b, top_label))
                
    return classification_gt, localization_gt

# -----------------------------
# Metric Calculation (Pairwise)
# -----------------------------
def calculate_metrics_pairwise(segs1, segs2, labels, tol, video_duration):
    # This logic reuses the robust time-based matrix from previous step
    # We reconstruct a simple matrix just for the pair to report Kappa and agreement percentage
    L = {lab: i for i, lab in enumerate(labels)}
    dim = len(labels)
    cm = np.zeros((dim, dim), dtype=float)
    
    # Use merged intervals just for this pair
    pair_intervals = get_combined_atomic_intervals([segs1, segs2], tol)
    
    for a, b in pair_intervals:
        l1 = get_label_at_interval(segs1, a, b, tol)
        l2 = get_label_at_interval(segs2, a, b, tol)
        cm[L[l1], L[l2]] += (b - a)
        
    annotated_end = float(cm.sum())
    
    # Trailing Silence Injection for Metrics
    if video_duration > annotated_end:
        tail = video_duration - annotated_end
        if NO_LABEL in labels:
            idx = labels.index(NO_LABEL)
            cm[idx, idx] += tail
    
    total_sec = float(cm.sum())
    agreed = float(np.trace(cm))
    pct = agreed / total_sec if total_sec > 0 else 0
    
    # Kappa
    total = cm.sum()
    pe = 0
    if total > 0:
        po = agreed / total
        row = cm.sum(axis=1) / total
        col = cm.sum(axis=0) / total
        pe = float((row * col).sum())
        kappa = (po - pe) / (1.0 - pe) if abs(1.0 - pe) > 1e-12 else 0
    else:
        kappa = 0
        
    return total_sec, annotated_end, pct, kappa, cm

# -----------------------------
# File Processing
# -----------------------------
def get_video_duration(video_path: Optional[str]) -> float:
    if not video_path or not os.path.exists(video_path): return 0.0
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps > 0: return frames / fps
    except:
        pass
    return 0.0

def process_file(eaf_path: str, video_dir: Optional[str], base_out_dir: str, tol: float):
    filename = os.path.basename(eaf_path)
    file_id = os.path.splitext(filename)[0]
    analysis_out_dir = os.path.join(base_out_dir, 'cross_annotation_analysis', file_id)
    eaf_out_dir = os.path.join(base_out_dir, 'processed_eaf')
    os.makedirs(analysis_out_dir, exist_ok=True)
    os.makedirs(eaf_out_dir, exist_ok=True)

    print(f"Processing: {filename}")
    eaf = pympi.Elan.Eaf(eaf_path)
    # Discover Groups (PID -> [Annotators])
    pid_groups = discover_annotator_groups(eaf)
    
    # Tracking Dictionary for our Custom Summary Report
    file_counts = {
        'annotators': collections.defaultdict(lambda: collections.defaultdict(int)),
        'agreed': collections.defaultdict(int)
    }
    
    if not pid_groups:
        print(f"  [WARN] No annotation groups found in {filename}")
        return [], 0.0, file_counts

    # Get Video Duration
    video_path = None
    if video_dir:
        base_name = os.path.splitext(filename)[0]
        cand = os.path.join(video_dir, f"{base_name}.mp4")
        if os.path.exists(cand):
            video_path = cand
    
    video_duration = get_video_duration(video_path)
    batch_rows = []
    
    # Process Each Participant
    for pid, annotators in pid_groups.items():
        # annotators is list of (Name, Tier)
        # Sort to ensure consistent pair ordering
        annotators.sort()
        
        # Gather segment counts per annotator
        for ann_name, tier_name in annotators:
            segs = tier_segments(eaf, tier_name)
            for s in segs:
                if s.label != NO_LABEL:
                    file_counts['annotators'][ann_name][s.label] += 1
        
        # GENERATE CONSENSUS TIERS
        class_gt, loc_gt = generate_consensus_tiers(pid, annotators, eaf, tol)
        
        # Gather agreed consensus counts
        for _, _, label in class_gt:
            if label != NO_LABEL:
                file_counts['agreed'][label] += 1
        
        # Write to EAF
        # Classification GT Tier
        gt_tier_name = f"AGREEMENT_{pid}_GT"
        if gt_tier_name in eaf.tiers: eaf.remove_tier(gt_tier_name)
        eaf.add_tier(gt_tier_name)
        for s, e, label in class_gt:
            eaf.add_annotation(gt_tier_name, int(s*1000), int(e*1000), label)
        
        # Localization GT Tier
        loc_tier_name = f"LOCALIZATION_{pid}_GT"
        if loc_tier_name in eaf.tiers: eaf.remove_tier(loc_tier_name)
        eaf.add_tier(loc_tier_name)
        for s, e in loc_gt:
            eaf.add_annotation(loc_tier_name, int(s*1000), int(e*1000), "MouthActive")

        # CALCULATE PAIRWISE METRICS
        # ---------------------------------------------
        # We compare every pair to see individual agreement

        if len(annotators) < 2: continue
        for (ann1, tier1), (ann2, tier2) in itertools.combinations(annotators, 2):
            segs1 = tier_segments(eaf, tier1)
            segs2 = tier_segments(eaf, tier2)
            all_labels = sorted(list(set([s.label for s in segs1] + [s.label for s in segs2] + [NO_LABEL])))
            
            total_sec, ann_end, pct, kappa, cm = calculate_metrics_pairwise(
                segs1, segs2, all_labels, tol, video_duration
            )
            
            # Save CSV Report
            csv_name = f"{pid}_{ann1}_vs_{ann2}.agreement.csv"
            with open(os.path.join(analysis_out_dir, csv_name), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["participant", pid])
                w.writerow(["pair", f"{ann1} vs {ann2}"])
                w.writerow(["video_duration", f"{total_sec:.2f}"])
                w.writerow(["kappa", f"{kappa:.4f}"])
                w.writerow([])
                w.writerow(["confusion_matrix_seconds"])
                w.writerow(["A1 (Row) \ A2 (Col)"] + all_labels)
                for i, row in enumerate(cm):
                    w.writerow([all_labels[i]] + [f"{x:.2f}" for x in row])

            # Store for batch summary
            batch_rows.append([filename, f"{pid} ({ann1}v{ann2})", total_sec, ann_end, pct, kappa])

    # Save modified EAF
    eaf.to_file(os.path.join(eaf_out_dir, f"{file_id}.with_GT.eaf"))

    # Return rows AND the processed video duration (once per file) for global stats
    return batch_rows, video_duration, file_counts

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--video_dir", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tolerance", type=float, default=0.1)
    ap.add_argument("--save_video", default=False, help="Whether to save videos with GT overlays (not implemented)")
    args = ap.parse_args()

    if args.video_dir:
        convert_mov_to_mp4(args.video_dir)
        standardize_filenames(args.video_dir, ('.mp4',))

    standardize_filenames(args.input_dir, ('.eaf',))
    eaf_files = glob.glob(os.path.join(args.input_dir, "*.eaf"))
    
    all_metrics = []
    total_processing_time_accumulated = 0.0
    
    # Store text summaries for our new report
    summary_report_lines = []
    global_agreed_counts = collections.defaultdict(int)
    
    # Target columns for summary
    target_labels = ["Mouthing", "MouthGesture", "Others"]

    for f in eaf_files:
        try:
            rows, vid_len, file_counts = process_file(f, args.video_dir, args.out_dir, args.tolerance)
            all_metrics.extend(rows)
            # Add to total time processed (logic: sum of file durations)
            if vid_len > 0:
                total_processing_time_accumulated += vid_len
            elif rows:
                # If video load failed, use annotated length from first pair as approximation (backup)
                total_processing_time_accumulated += rows[0][3] 
                
            # Build Summary Report for this file
            filename_only = os.path.basename(f)
            summary_report_lines.append(f"File name: {filename_only}")
            
            # Formatting the header dynamically
            header = f"{'class label -->':<20}" + "".join([f"{lbl:>15}" for lbl in target_labels])
            summary_report_lines.append(header)
            
            # Print Annotator Counts
            for ann_name, counts in file_counts['annotators'].items():
                row_str = f"{ann_name:<20}" + "".join([f"{counts.get(lbl, 0):>15}" for lbl in target_labels])
                summary_report_lines.append(row_str)
            
            # Print Agreed Counts
            agreed = file_counts['agreed']
            agreed_str = f"{'agreed count':<20}" + "".join([f"{agreed.get(lbl, 0):>15}" for lbl in target_labels])
            summary_report_lines.append(agreed_str)
            summary_report_lines.append("") # Blank line separator
            
            # Accumulate global totals
            for lbl in target_labels:
                global_agreed_counts[lbl] += agreed.get(lbl, 0)
                
        except Exception as e:
            print(f"Error processing {f}: {e}")

    # Build Global Summary Totals
    summary_report_lines.append("Total agreed count:")
    total_agreed_str = "".join([f"{lbl}: {global_agreed_counts[lbl]}   " for lbl in target_labels])
    summary_report_lines.append(total_agreed_str)
    
    # Join into a single text block
    full_summary_text = "\n".join(summary_report_lines)

    # Write Batch Summary
    if all_metrics:
        b_path = os.path.join(args.out_dir, 'cross_annotation_analysis', "batch_summary.csv")
        with open(b_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "pair_id", "video_sec", "annotated_end", "pct_agreement", "kappa"])
            for r in all_metrics:
                w.writerow([r[0], r[1], f"{r[2]:.2f}", f"{r[3]:.2f}", f"{r[4]:.4f}", f"{r[5]:.4f}"])

        # Write Overall Metrics
        # Correctly average the kappas/pcts from the pairs
        avg_pct = sum([x[4] for x in all_metrics]) / len(all_metrics)
        avg_kap = sum([x[5] for x in all_metrics]) / len(all_metrics)
        
        o_path = os.path.join(args.out_dir, 'cross_annotation_analysis', "overall_metrics.csv")
        with open(o_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerow(["total_files", len(eaf_files)])
            w.writerow(["total_pairs_analyzed", len(all_metrics)])
            w.writerow(["total_video_minutes_processed", f"{total_processing_time_accumulated/60:.4f}"])
            w.writerow(["total_video_seconds_processed", f"{total_processing_time_accumulated:.2f}"])
            w.writerow(["average_pair_agreement", f"{avg_pct:.4f}"])
            w.writerow(["average_pair_kappa", f"{avg_kap:.4f}"])
        
        # --- WRITE AND PRINT THE NEW COUNT SUMMARY ---
        summary_path = os.path.join(args.out_dir, 'cross_annotation_analysis', "count_summary.txt")
        with open(summary_path, "w") as f:
            f.write(full_summary_text)

        print("\n" + "="*50)
        print("                 COUNT SUMMARY")
        print("="*50)
        print(full_summary_text)
        print("="*50)
        
        print(f"\nDone. Processed {len(eaf_files)} files.")
        print(f"Total Content Duration: {total_processing_time_accumulated:.2f} seconds")
        print(f"Average Kappa: {avg_kap:.4f}")

if __name__ == "__main__":
    main()