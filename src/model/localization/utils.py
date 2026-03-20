#############################################################################
###  adopted from Tim Henrik Sandermann implementation at NII internship  ###
#############################################################################

from datetime import datetime
import math
from collections import defaultdict
from torch.utils.data import Dataset
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
import torch.nn as nn
from torch_geometric.nn import GraphConv
import pympi.Elan
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.cm as cm
import pickle
import os
from scipy.spatial import KDTree
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import random
import json
from pathlib import Path
import re


class MouthMovementEvaluator:
    def __init__(self, ground_truth, predictions, prediction_proba, segment_keys, fps=60):
        self.gt = list(ground_truth)
        self.pred = list(predictions)
        self.proba = list(prediction_proba)
        self.segment_keys = list(segment_keys)
        self.fps = fps

        self.class_colors = self.set_class_colors()

    def evaluate(self):
        print(print(classification_report(self.gt, self.pred)))

    def plot(self):
        # Each of the samples has a segment key with the following structure X_X_<start-time-s>_<end_time_s>_<class>
        band_duration = 60.0  # seconds per time window

        # Group data by X1_X2
        grouped_data = defaultdict(list)
        for key, gt_label, pred_label in zip(self.segment_keys, self.gt, self.pred):
            x1x2, start, end, _, _ = parse_segment_key(key)
            grouped_data[x1x2].append((start, end, int(gt_label), int(pred_label)))

        for x1x2, segments in grouped_data.items():
            # Determine time span
            min_time = min(s for s, _, _, _ in segments)
            max_time = max(e for _, e, _, _ in segments)
            num_bands = math.ceil((max_time - min_time) / band_duration)

            # Create subplots (one per 60s window)
            fig, axs = plt.subplots(num_bands, 1, figsize=(12, 2.5 * num_bands), sharey=False)

            if num_bands == 1:
                axs = [axs]  # Ensure it's always a list

            for band_idx in range(num_bands):
                band_start = band_idx * band_duration
                band_end = (band_idx + 1) * band_duration
                ax = axs[band_idx]

                yticks = [1, 0]
                yticklabels = ['GT', 'Pred']

                # Plot GT
                for start, end, gt_label, _ in segments:
                    if band_start <= start < band_end:
                        ax.barh(y=1, width=end - start, left=start, height=0.4,
                                color=self.class_colors[gt_label], edgecolor='black')

                # Plot Prediction
                for start, end, _, pred_label in segments:
                    if band_start <= start < band_end:
                        ax.barh(y=0, width=end - start, left=start, height=0.4,
                                color=self.class_colors[pred_label], edgecolor='black')

                ax.set_xlim(band_start, band_end)
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticklabels)
                ax.set_xlabel("Time (s)")
                ax.set_title(f"{x1x2} | {int(band_start)}–{int(band_end)}s")
                ax.grid(True, axis='x')

            # Legend (placed in last subplot)
            unique_classes = sorted(set(self.gt + self.pred))
            handles = [mpatches.Patch(color=self.class_colors[cls], label=str(cls))
                       for cls in unique_classes]
            axs[-1].legend(handles=handles, loc='upper right')

            plt.tight_layout()
            plt.show()

        ConfusionMatrixDisplay.from_predictions(self.gt, self.pred)
        plt.show()

    def print_mismatched_intervals(self):
        """
        Returns a list of mismatched segments as tuples:
        (x1x2, start_time, end_time, ground_truth_class, predicted_class)
        Ordered by the most confident mismatch first.
        """
        mismatches = []

        for key, gt_label, pred_label, proba in zip(self.segment_keys, self.gt, self.pred, self.proba):
            if gt_label != pred_label:
                x1x2, start, end, _ = parse_segment_key(key)

                start_min, start_sec = divmod(start, 60)
                end_min, end_sec = divmod(end, 60)

                start_str = f"{int(start_min):02d}:{start_sec:05.2f}"
                end_str = f"{int(end_min):02d}:{end_sec:05.2f}"

                mismatches.append((proba[pred_label], x1x2, start_str, end_str, int(gt_label), int(pred_label)))

        mismatches.sort(reverse=True, key=lambda x: x[0])

        for m in mismatches:
            print(f"Mismatch in {m[1]} from {m[2]}s to {m[3]}s: GT={m[4]}, Pred={m[5]}. Probability={m[0]}")

    def set_class_colors(self):
        classes = set(int(cls) for cls in self.gt + self.pred)
        cmap = cm.get_cmap('tab20', len(classes))
        return {cls: cmap(i) for i, cls in enumerate(classes)}


class MouthMovementDetector:
    def __init__(self):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self, X, y):
        pass

    def predict(self, X):
        pass

    def predict_boundaries(self, X, keys, class_encoder, save_path):
        pass


class StandardMouthMovementDetector(MouthMovementDetector):
    # Can be used for standard sklearn classifiers as RF, LR, ...
    def __init__(self, model_path):
        super().__init__()

        self.model_path = model_path

        if self.model_path is not None:
            self.load_model(model_path)
        else:
            self.create_model()

    def load_model(self, model_path):
        # Load pickle file model (.pkl)
        with open('model.pkl', 'rb') as f:
            self.model = pickle.load(f)

    def create_model(self):
        pass

    def train(self, X, y):
        if self.model_path is not None:
            print("Warning: Model was loaded, skip training.")
        else:
            self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X), self.model.predict_proba(X).max(axis=1)

    def predict_boundaries(self, X, keys, class_encoder, save_path=None, elan_folder=None):
        # Predict boundaries as well as movement and non-movement
        # Save this as .json such that one can map this to an annotation in the ELAN software
        prediction_meta = {}
        reverse_class_encoder = {v: k for k, v in class_encoder.items()}

        (pred, prob) = self.predict(X)

        for cur_pred, cur_prob, cur_key in zip(pred, prob, keys):
            x1x2, start, end, _, video = parse_segment_key(cur_key)

            key = x1x2 + video

            if key not in prediction_meta:
                prediction_meta[key] = []

            prediction_meta[key].append({
                "start_time": start,
                "end_time": end,
                "label": reverse_class_encoder[cur_pred],
                "confidence": cur_prob
            })
        
        logging = []
        # Sort video labels by start time
        for vid in prediction_meta:
            prediction_meta[vid].sort(key=lambda x: x["start_time"])

            # Fuse consecutive segments with the same label
            fused = []
            for segment in prediction_meta[vid]:
                if not fused:
                    fused.append(segment)
                else:
                    last = fused[-1]
                    if last["label"] == segment["label"]:
                        # Extend the last segment
                        last["end_time"] = max(last["end_time"], segment["end_time"])
                    else:
                        fused.append(segment)
            prediction_meta[vid] = fused

            # If elan_folder is provided, calculate IoU of predictions with existing ELAN annotations
            if elan_folder is not None:
                folder_path = Path(elan_folder)
                for fname in folder_path.rglob("*.eaf"):
                    if fname.parts[-3].startswith("0"):
                        if extract_video_code(vid) in fname.name.replace("-", "_"):
                            eaf = pympi.Elan.Eaf(fname)
                            for tier_id, tier_data in eaf.tiers.items():
                                if (('MouthAction' not in tier_id) and ('Mouth' not in tier_id)) or ('(roman)' in tier_id):
                                    continue

                                # Extract annotation intervals (start, end) from ELAN tier
                                annotations = []
                                for ann_id in tier_data[0]:
                                    start, end, value, _ = tier_data[0][ann_id]
                                    start_time = eaf.timeslots[start] / 1000  # ms to seconds
                                    end_time = eaf.timeslots[end] / 1000

                                    annotations.append((start_time, end_time))

                                def merge_intervals(intervals):
                                    """Merge overlapping intervals into a union of disjoint intervals."""
                                    if not intervals:
                                        return []
                                    intervals = sorted(intervals, key=lambda x: x[0])
                                    merged = [intervals[0]]
                                    for start, end in intervals[1:]:
                                        last_start, last_end = merged[-1]
                                        if start <= last_end:  # overlap
                                            merged[-1] = (last_start, max(last_end, end))
                                        else:
                                            merged.append((start, end))
                                    return merged


                                def union_length(intervals):
                                    """Total length of merged intervals (union coverage)."""
                                    return sum(end - start for start, end in merge_intervals(intervals))

                                def iou_over_unions(pred_intervals, ann_intervals):
                                    """IoU between union of prediction intervals and union of annotation intervals,
                                    restricted to the span from the first to the last annotation."""
                                    if not pred_intervals or not ann_intervals:
                                        return 0.0

                                    # Restrict evaluation range to [first_ann_start, last_ann_end]
                                    ann_start = min(start for start, _ in ann_intervals)
                                    ann_end = max(end for _, end in ann_intervals)
                                    eval_range = (ann_start, ann_end)

                                    def clip(intervals, clip_range):
                                        c_start, c_end = clip_range
                                        clipped = []
                                        for s, e in intervals:
                                            s_clipped = max(s, c_start)
                                            e_clipped = min(e, c_end)
                                            if s_clipped < e_clipped:
                                                clipped.append((s_clipped, e_clipped))
                                        return clipped

                                    def merge(intervals):
                                        if not intervals:
                                            return []
                                        intervals = sorted(intervals, key=lambda x: x[0])
                                        merged = [intervals[0]]
                                        for s, e in intervals[1:]:
                                            ls, le = merged[-1]
                                            if s <= le:
                                                merged[-1] = (ls, max(le, e))
                                            else:
                                                merged.append((s, e))
                                        return merged

                                    def union_length(intervals):
                                        return sum(e - s for s, e in merge(intervals))

                                    # Clip predictions and annotations to evaluation range
                                    pred_clipped = merge(clip(pred_intervals, eval_range))
                                    ann_clipped = merge(clip(ann_intervals, eval_range))

                                    # Compute intersection
                                    intersection = 0
                                    for ps, pe in pred_clipped:
                                        for as_, ae in ann_clipped:
                                            inter_start = max(ps, as_)
                                            inter_end = min(pe, ae)
                                            intersection += max(0, inter_end - inter_start)

                                    union = union_length(pred_clipped) + union_length(ann_clipped) - intersection
                                    return intersection / union if union > 0 else 0.0

                                
                                # --- Case 1: MOVEMENT ---
                                pred_intervals = [
                                    (pred["start_time"], pred["end_time"])
                                    for pred in prediction_meta[key]
                                    if pred["label"] == "movement"
                                ]

                                iou_score = iou_over_unions(pred_intervals, annotations)
                                logging.append((fname.name, iou_score))
                                """
                                # --- Case 2: NO-MOVEMENT ---
                                    elif pred["label"] == "'no-mouth-movement'":
                                        # IoU is defined against the *complement* of annotations
                                        # First build "no-annotation" intervals between annotated regions
                                        no_ann_intervals = []
                                        last_end = 0
                                        for ann_start, ann_end in sorted(annotations):
                                            if ann_start > last_end:
                                                no_ann_intervals.append((last_end, ann_start))
                                            last_end = max(last_end, ann_end)
                                        # Optionally add open-ended interval after last annotation
                                        # (depends if your video ends at known time T)
                                        # no_ann_intervals.append((last_end, video_duration))

                                        best_iou = 0.0
                                        for gap_start, gap_end in no_ann_intervals:
                                            iou = interval_iou(pred["start_time"], pred["end_time"], gap_start, gap_end)
                                            best_iou = max(best_iou, iou)
                                        print(f"NO-MOVEMENT IoU for {pred['start_time']}–{pred['end_time']}: {best_iou:.3f}")
                                """
        # Filter logging for best score per video
        if elan_folder is not None:
            best_scores = {}
            for fname, score in logging:
                if fname not in best_scores or score > best_scores[fname]:
                    best_scores[fname] = score

            # Print results
            for fname, score in best_scores.items():
                print(f"Best IoU for {fname}: {score:.3f}")
        # Save to json file 
        if save_path is not None:
            with open(save_path, 'w') as f:
                json.dump(prediction_meta, f, indent=4)

        return prediction_meta


class GCNMouthMovementDetector(MouthMovementDetector):
    def __init__(self, model_path=None, in_channels=None, batching=True):
        super().__init__()

        self.model_path = model_path
        self.batching = batching

        assert in_channels is not None, "in_channels must be specified if model_path is None."
        self.model = MouthSTGCN(in_channels=in_channels).to(device=self.device)

        if self.model_path is not None:
            self.load_model(model_path)
        
        self.crit = nn.BCELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, train_loader, test_loader, epochs=10):
        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = self._run_epoch_batched(train_loader, train=True) if self.batching else self._run_epoch(train_loader, train=True)
            val_loss, val_acc = self._run_epoch_batched(test_loader, train=True) if self.batching else self._run_epoch(test_loader, train=True)
            print(f"E{epoch:02d}: train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {val_loss:.4f} acc {val_acc:.3f}")
    
    def _run_epoch(self, loader, train: bool):
        if train:
            self.model.train()
        else:
            self.model.eval()

        loss_sum = 0.0
        y_preds = []
        y_trues = []
        
        for data in loader:
            data = data.to(self.device)
            if train:
                self.opt.zero_grad()
            out = self.model(data.x.unsqueeze(0), data.edge_index)
            loss = self.crit(out, data.y.view(1).float())
            if train:
                loss.backward()
                self.opt.step()
            loss_sum += loss.item()
            preds = (out > 0.5).long()

            y_preds.append(preds.item())
            y_trues.append(data.y.item())

        return loss_sum / len(loader), accuracy_score(y_trues, y_preds)

    def _run_epoch_batched(self, loader, train: bool):
        if train:
            self.model.train()
        else:
            self.model.eval()

        loss_sum = 0.0
        y_preds = []
        y_trues = []

        for batch in loader:
            batch = batch.to(self.device)
            
            # batch.x should already be shaped (B, T, N, C)
            out = self.model(batch.x, batch.edge_index)  # → (B,)
            loss = self.crit(out, batch.y.float())       # BCELoss needs float targets

            if train:
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            loss_sum += loss.item()

            preds = (out > 0.5).long()
            y_preds.extend(preds.tolist())
            y_trues.extend(batch.y.tolist())

        return loss_sum / len(loader), accuracy_score(y_trues, y_preds)

    def predict(self, loader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                out = self.model(data.x, data.edge_index)
                preds = (out > 0.5).long()
                predictions.append(preds.cpu().numpy())
        return np.concatenate(predictions, axis=0)
    
    def predict_boundaries(self, loader, class_encoder, save_path=None, elan_folder=None):
        prediction_meta = {}
        reverse_class_encoder = {v: k for k, v in class_encoder.items()}

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                prob_batch = self.model(batch.x, batch.edge_index)  # (B,) values in [0,1]

                for cur_prob, cur_segment_key in zip(prob_batch.cpu(), batch.segment_key):
                    cur_prob = float(cur_prob)
                    cur_pred = int(cur_prob > 0.5)
                    conf = cur_prob if cur_pred == 1 else 1.0 - cur_prob

                    x1x2, start, end, _, video = parse_segment_key(cur_segment_key)
                    key = x1x2 + video

                    if key not in prediction_meta:
                        prediction_meta[key] = []

                    prediction_meta[key].append({
                        "start_time": start,
                        "end_time": end,
                        "label": reverse_class_encoder[cur_pred],
                        "confidence": conf
                    })

        logging = []
        # Sort video labels by start time
        for vid in prediction_meta:
            prediction_meta[vid].sort(key=lambda x: x["start_time"])

            # Fuse consecutive segments with the same label
            fused = []
            for segment in prediction_meta[vid]:
                if not fused:
                    fused.append(segment)
                else:
                    last = fused[-1]
                    if last["label"] == segment["label"]:
                        # Extend the last segment
                        last["end_time"] = max(last["end_time"], segment["end_time"])
                    else:
                        fused.append(segment)
            prediction_meta[vid] = fused

            # If elan_folder is provided, calculate IoU of predictions with existing ELAN annotations
            if elan_folder is not None:
                folder_path = Path(elan_folder)
                for fname in folder_path.rglob("*.eaf"):
                    if fname.parts[-3].startswith("0"):
                        if extract_video_code(vid) in fname.name.replace("-", "_"):
                            eaf = pympi.Elan.Eaf(fname)
                            for tier_id, tier_data in eaf.tiers.items():
                                if (('MouthAction' not in tier_id) and ('Mouth' not in tier_id)) or ('(roman)' in tier_id):
                                    continue

                                # Extract annotation intervals (start, end) from ELAN tier
                                annotations = []
                                for ann_id in tier_data[0]:
                                    start, end, value, _ = tier_data[0][ann_id]
                                    start_time = eaf.timeslots[start] / 1000  # ms to seconds
                                    end_time = eaf.timeslots[end] / 1000

                                    annotations.append((start_time, end_time))

                                def merge_intervals(intervals):
                                    """Merge overlapping intervals into a union of disjoint intervals."""
                                    if not intervals:
                                        return []
                                    intervals = sorted(intervals, key=lambda x: x[0])
                                    merged = [intervals[0]]
                                    for start, end in intervals[1:]:
                                        last_start, last_end = merged[-1]
                                        if start <= last_end:  # overlap
                                            merged[-1] = (last_start, max(last_end, end))
                                        else:
                                            merged.append((start, end))
                                    return merged


                                def union_length(intervals):
                                    """Total length of merged intervals (union coverage)."""
                                    return sum(end - start for start, end in merge_intervals(intervals))

                                def iou_over_unions(pred_intervals, ann_intervals):
                                    """IoU between union of prediction intervals and union of annotation intervals,
                                    restricted to the span from the first to the last annotation."""
                                    if not pred_intervals or not ann_intervals:
                                        return 0.0

                                    # Restrict evaluation range to [first_ann_start, last_ann_end]
                                    ann_start = min(start for start, _ in ann_intervals)
                                    ann_end = max(end for _, end in ann_intervals)
                                    eval_range = (ann_start, ann_end)

                                    def clip(intervals, clip_range):
                                        c_start, c_end = clip_range
                                        clipped = []
                                        for s, e in intervals:
                                            s_clipped = max(s, c_start)
                                            e_clipped = min(e, c_end)
                                            if s_clipped < e_clipped:
                                                clipped.append((s_clipped, e_clipped))
                                        return clipped

                                    def merge(intervals):
                                        if not intervals:
                                            return []
                                        intervals = sorted(intervals, key=lambda x: x[0])
                                        merged = [intervals[0]]
                                        for s, e in intervals[1:]:
                                            ls, le = merged[-1]
                                            if s <= le:
                                                merged[-1] = (ls, max(le, e))
                                            else:
                                                merged.append((s, e))
                                        return merged

                                    def union_length(intervals):
                                        return sum(e - s for s, e in merge(intervals))

                                    # Clip predictions and annotations to evaluation range
                                    pred_clipped = merge(clip(pred_intervals, eval_range))
                                    ann_clipped = merge(clip(ann_intervals, eval_range))

                                    # Compute intersection
                                    intersection = 0
                                    for ps, pe in pred_clipped:
                                        for as_, ae in ann_clipped:
                                            inter_start = max(ps, as_)
                                            inter_end = min(pe, ae)
                                            intersection += max(0, inter_end - inter_start)

                                    union = union_length(pred_clipped) + union_length(ann_clipped) - intersection
                                    return intersection / union if union > 0 else 0.0

                                
                                # --- Case 1: MOVEMENT ---
                                pred_intervals = [
                                    (pred["start_time"], pred["end_time"])
                                    for pred in prediction_meta[key]
                                    if pred["label"] == "movement"
                                ]

                                iou_score = iou_over_unions(pred_intervals, annotations)
                                logging.append((fname.name, iou_score))
                                """
                                # --- Case 2: NO-MOVEMENT ---
                                    elif pred["label"] == "'no-mouth-movement'":
                                        # IoU is defined against the *complement* of annotations
                                        # First build "no-annotation" intervals between annotated regions
                                        no_ann_intervals = []
                                        last_end = 0
                                        for ann_start, ann_end in sorted(annotations):
                                            if ann_start > last_end:
                                                no_ann_intervals.append((last_end, ann_start))
                                            last_end = max(last_end, ann_end)
                                        # Optionally add open-ended interval after last annotation
                                        # (depends if your video ends at known time T)
                                        # no_ann_intervals.append((last_end, video_duration))

                                        best_iou = 0.0
                                        for gap_start, gap_end in no_ann_intervals:
                                            iou = interval_iou(pred["start_time"], pred["end_time"], gap_start, gap_end)
                                            best_iou = max(best_iou, iou)
                                        print(f"NO-MOVEMENT IoU for {pred['start_time']}–{pred['end_time']}: {best_iou:.3f}")
                                """
        # Filter logging for best score per video
        if elan_folder is not None:
            best_scores = {}
            for fname, score in logging:
                if fname not in best_scores or score > best_scores[fname]:
                    best_scores[fname] = score

            # Print results
            for fname, score in best_scores.items():
                print(f"Best IoU for {fname}: {score:.3f}")

        # Save to json file 
        if save_path is not None:
            with open(save_path, 'w') as f:
                json.dump(prediction_meta, f, indent=4)

        return prediction_meta

    def save_model(self, save_path):
        if self.model is not None:
            torch.save(self.model.state_dict(), save_path)
        else:
            raise ValueError("Model is not trained or loaded.")
    
    def load_model(self, model_path):
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
        else:
            raise FileNotFoundError(f"Model file {model_path} does not exist.")

class MouthGraphDataset(Dataset):
    def __init__(self, samples, label_encoder=None, shuffle=True):
        self.samples = samples

        if shuffle:
            random.shuffle(self.samples)  # Shuffle samples for better training (avoid the sequences based on preprocessing)

        # self.first_sample_coords = samples[0]['landmarks_cheeks'] + samples[0]['landmarks_mouth_inner']
        self.first_sample_coords = np.concatenate([np.array(samples[0]['landmarks_cheeks']), 
                                                   np.array(samples[0]['landmarks_mouth_inner'])], axis=1)[0]

        self.label_encoder = label_encoder

        self.edges_idx = self.build_edges()
    
    def build_edges(self):
        edges = set()
        tree = KDTree(self.first_sample_coords)
        for i, coord in enumerate(self.first_sample_coords):
            # Find the nearest neighbors within a certain distance
            indices = tree.query(coord, k=5)[1]

            for j in indices:
                if i != j:
                    edge = (min(i, j), max(i, j))
                    edges.add(edge)
        
        return torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        # x = item['landmarks_cheeks'] + item['landmarks_mouth_inner']
        x = np.concatenate([item['landmarks_cheeks'], item['landmarks_mouth_inner']], axis=1)
        y = self.label_encoder[item['label']]
        segment_key = item['segment_key']

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        return Data(x=x, y=y, edge_index=self.edges_idx, segment_key=segment_key)
    
    def __len__(self):
        return len(self.samples)

class STBlock(nn.Module):
    """Spatial graph conv → temporal 1‑D conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels)
        self.tcn = nn.Conv2d(out_channels, out_channels, (1, 3), padding=(0, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, edge_index):
        # x: (B, T, N, C)
        b, t, n, c = x.shape
        x = x.reshape(b * t, n, c)
        x = self.gcn(x, edge_index).view(b, t, n, -1)
        x = x.permute(0, 3, 2, 1)           # (B,C,N,T) for Conv2d
        x = self.act(self.bn(self.tcn(x)))
        x = x.permute(0, 3, 2, 1)           # back to (B,T,N,C)
        return x

class MouthSTGCN(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64):
        super().__init__()
        self.block1 = STBlock(in_channels, hidden)
        self.block2 = STBlock(hidden, hidden)
        self.block3 = STBlock(hidden, hidden)  

        self.bn = nn.BatchNorm1d(hidden)
        self.ln = nn.LayerNorm(hidden)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        x = self.block1(x, edge_index)
        x = self.block2(x, edge_index)
        x = self.block3(x, edge_index)  # Reuse block2 for deeper representation

        x = x.mean(dim=(1, 2))        # safe reshape

        # Apply BatchNorm1d and Dropout
        #x = self.bn(x)

        x = self.ln(x)
        x = self.dropout(x)

        return torch.sigmoid(self.fc(x)).squeeze(-1)

def parse_segment_key(key):
    # Parses "X1_X2_12.0_18.5_classname" -> ("X1_X2", 12.0, 18.5, "classname")
    parts = key.split('_')
    x1x2 = "_".join(parts[:2])
    start = float(parts[2])
    end = float(parts[3])
    label = "_".join(parts[4:]).split('#')[0]
    video = "_".join(parts[4:]).split('#')[-1]

    return x1x2, start, end, label, video

def prediction_to_elan(prediction_dict, elan_folder, classifier_name, window_len, stride, conf_threshold=0.5, use_confidence=True, new_file=True, low_conf_suffix="(LC)", move_only=True):
    assert os.path.isdir(elan_folder)

    if not isinstance(prediction_dict, list):
        prediction_dict = [prediction_dict]
    
    loaded_dicts = []
    for entry in prediction_dict:
        if isinstance(entry, str) and entry.endswith(".json"):
            with open(entry, 'r') as f:
                loaded_dicts.append(json.load(f))
        elif isinstance(entry, dict):
            loaded_dicts.append(entry)
        else:
            raise ValueError("Entries must be either dicts or paths to .json files")

    folder_path = Path(elan_folder)
    for fname in folder_path.rglob("*.eaf"):
        if fname.parts[-3].startswith("0"):
            pred_for_file = False

            base = os.path.splitext(fname.name)[0]
            eaf_path = fname
            # Matched video and annotation
            eaf = pympi.Elan.Eaf(eaf_path)

            for pred_source, classifier in zip(loaded_dicts, classifier_name):
                for dict_key in pred_source:
                    if (base in dict_key) or (base.replace("-", "_") in dict_key):
                        tier_id = f"{dict_key.split('_')[0]}_MouthBound_{classifier}_{os.path.splitext(os.path.basename(fname.name))[0]}"
                        assert tier_id not in eaf.get_tier_names(), "Tier already safed in elan file."

                        eaf.add_tier(tier_id)

                        annotations = pred_source[dict_key]
                        if not annotations:
                            continue

                        pred_for_file = True

                        # Determine end time 
                        max_end_time = max(ann["end_time"] for ann in annotations)
                        num_bins = math.ceil(max_end_time / stride)
                        bin_votes = [defaultdict(float) for _ in range(num_bins)]

                        # Voting per bin
                        for ann in annotations:
                            label = ann["label"]
                            start_bin = int(ann["start_time"] // stride)
                            end_bin = int(ann["end_time"] // stride)
                            weight = ann["confidence"] if use_confidence else 1.0

                            for bin_idx in range(start_bin, end_bin + 1):
                                if bin_idx < len(bin_votes):
                                    bin_votes[bin_idx][label] += weight
                        
                        # Merge adjacent bins with same label
                        prev_label = None
                        start_bin = None

                        # Create annotations based on voting
                        for bin_idx, votes in enumerate(bin_votes):
                            if not votes:
                                continue

                            # Find the label with the highest vote
                            label_counts = votes.items()
                            sorted_labels = sorted(label_counts, key=lambda x: (-x[1], x[0]))  # highest weight first
                            top_label, top_score = sorted_labels[0]
                            total_score = sum(votes.values())

                            # Check if there is a tie
                            if len(sorted_labels) > 1 and sorted_labels[1][1] == top_score:
                                label = f"unknown {low_conf_suffix}"
                            else:
                                label = top_label
                                if (top_score / total_score) < conf_threshold:
                                    label += f" {low_conf_suffix}"

                            if label == prev_label:
                                # Continue the current annotation
                                continue
                            else:
                                if prev_label is not None:
                                    start_time = int(start_bin * stride * 1000)
                                    end_time = int(bin_idx * stride * 1000)

                                    if not move_only or (move_only and prev_label != "no-mouth-movement"):
                                        eaf.add_annotation(tier_id, start_time, end_time, prev_label)

                                # Start new segment
                                prev_label = label
                                start_bin = bin_idx
                            
                        # Final annotation
                        if prev_label is not None and start_bin is not None:
                            start_time = int(start_bin * stride * 1000)
                            end_time = int((bin_idx + 1) * stride * 1000)
                            eaf.add_annotation(tier_id, start_time, end_time, prev_label)

        if pred_for_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if new_file else ""
            folder_path = os.path.join(elan_folder, "elan_with_predictions")
            os.makedirs(folder_path, exist_ok=True)

            eaf.to_file(os.path.join(folder_path, base + timestamp + ".eaf"))

def prediction_to_elan_no_overlap(prediction_dict, elan_folder, classifier_name, use_confidence, new_file=True):
    assert os.path.isdir(elan_folder)

    list = []
    for fname in os.listdir(elan_folder):
        if fname.endswith(".mp4"):
            base = os.path.splitext(fname)[0]
            eaf_path = os.path.join(elan_folder, base + ".eaf")

            for dict_key in prediction_dict:
                if base in dict_key:
                    # Matched video and annotation
                    eaf = pympi.Elan.Eaf(eaf_path)

                    tier_id = f"{dict_key.split('_')[0]}_MouthBound_{classifier_name}"
                    assert tier_id not in eaf.get_tier_names(), "Tier already safed in elan file."

                    eaf.add_tier(tier_id)

                    for ann in prediction_dict[dict_key]:
                        start_ms = int(ann["start_time"] * 1000)
                        end_ms = int(ann["end_time"] * 1000)
                        label = ann["label"]

                        eaf.add_annotation(tier_id, start_ms, end_ms, label)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if new_file else ""
                    eaf.to_file(os.path.join(elan_folder, base + timestamp + ".eaf"))

                    break

def collate_graph_windows(batch):
    xs, ys, ses = [], [], []
    for sample in batch:
        xs.append(sample.x.unsqueeze(0))  # Add batch dimension
        ys.append(sample.y)
        ses.append(sample.segment_key)

    x = torch.cat(xs, dim=0)  # Concatenate along batch dimension
    y = torch.stack(ys, dim=0)  # Stack labels

    edge_index = batch[0].edge_index  # All samples should have the same edge index

    return Data(x=x, y=y, edge_index=edge_index, segment_key=ses)

def find_last_moving_annotation(data_folder):
    # Contains folders with .json files in them. These json files store annotations. 
    # The json file has the form of a dict, where the dict entries are strings of the form <person_id>_<agesex>_<start_time>_<end_time>_<label>
    # Search for the ones with the label "mouth-movement" and print the subfolder name as well as the start_time of the last one.
    for subfolder in os.listdir(data_folder):
        subfolder_path = os.path.join(data_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        for json_file in os.listdir(subfolder_path):
            if json_file.endswith(".json"):
                json_path = os.path.join(subfolder_path, json_file)
                with open(json_path, 'r') as f:
                    annotations = json.load(f)

                last_moving_time = 0
                for key, ann in annotations.items():
                    key_components = key.split('_')
                    label = key_components[-1]  # Assuming the label is the last component
                    start_time = float(key_components[2])
                    if label == "movement":
                        last_moving_time = max(last_moving_time, start_time)

                if last_moving_time > 0:
                    print(f"Subfolder: {subfolder}, Last moving annotation start time: {last_moving_time}")


def extract_video_code(s, codes=("NS", "IS", "TY", "FO")):
    # Build regex to match any of the two-letter codes
    pattern = re.compile(r'(' + "|".join(codes) + r').*')
    match = pattern.search(s)
    return match.group(0) if match else None