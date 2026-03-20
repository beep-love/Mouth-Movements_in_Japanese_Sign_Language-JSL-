import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import random
from collections import defaultdict
from datetime import datetime, timedelta

class FaceBodySequenceDataset(Dataset):
    def __init__(self, json_path, root_dir, task='classification', face_transform=None, body_transform=None, label_mapping=None,
                 return_landmarks=True, mode='both', min_seq_len=None, max_seq_len=None, annotation_end_info=None):
        """
        Args:
            json_path (str): Path to the metadata JSON file.
            root_dir (str): Directory containing face/ and body/ folders.
            task (str): 'classification' or 'localization'. Filters which segments to load.
            face_transform (callable): Face image transforms.
            body_transform (callable): Body image transforms.
            label_mapping (dict): Maps string labels to int.
            return_landmarks (bool): If True, returns normalized landmarks.
            mode (str): 'body', 'face', 'both', or 'no_img' (loads only landmarks).
        """
        assert mode in ['body', 'face', 'both', 'no_img'], f"Invalid mode: {mode}"
        assert task in ['classification', 'localization'], f"Invalid task: {task}"
        
        self.mode = mode
        self.task = task
        self.root_dir = root_dir
        
        # --- SEPARATED TRANSFORMS WITH SMART DEFAULTS ---
        self.face_transform = face_transform or transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor()
        ])
        
        self.body_transform = body_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.return_landmarks = return_landmarks
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len

        with open(json_path, 'r') as f:
            self.data = json.load(f)

        video_name = os.path.basename(json_path).split('.')[0]
        self.samples = []

        # --- Optional End Time Filtering for localization task --- (Not used) - 
        # Use when we want to exclude segments after a certain time point in the video (e.g., annotator stopped at 15 mins)
        # annotation_end_info is a json object that looks like:
        # {
        #     "Metadata": {
        #         "VID_001": "00:15:30",  # Annotator stopped at 15 mins 30 secs
        #         "VID_002": "00:45:00"
        #     },
        #     "Processed": {
        #         "VID_001": 900.0        # Another hard limit at 900 seconds (15 mins)
        #     }
        # }
        end_time_s = float('inf')
        if annotation_end_info is not None and (video_name in annotation_end_info.get('Metadata', {}) or video_name in annotation_end_info.get('Processed', {})):
            if video_name in annotation_end_info.get('Metadata', {}):
                time_obj = datetime.strptime(annotation_end_info['Metadata'][video_name], "%H:%M:%S")
                end_time_s = timedelta(hours=time_obj.hour, minutes=time_obj.minute, seconds=time_obj.second).total_seconds()
            if video_name in annotation_end_info.get('Processed', {}):
                end_time_s = min(annotation_end_info['Processed'][video_name], end_time_s)

        # --- Process JSON Entries ---
        for segment_key, frames in self.data.items():
            # Filter out empty frames (caused by missing faces)
            if not frames:
                continue

            # Task Filtering
            if not segment_key.startswith(self.task):
                continue

            # Extract label from unified key: task_pid_start_end_label_uuid
            parts = segment_key.split('_')
            raw_label = parts[-2].strip() 
            
            try:
                start_time = float(parts[-4]) 
            except ValueError:
                start_time = 0.0

            if end_time_s is not None and start_time > end_time_s:
                continue

            # Clean label
            label_str = raw_label.split()[0]
            
            # Extract Paths
            if self.mode == 'no_img':
                face_paths, body_paths = [], []
            else:
                face_paths = [entry['face_path'] for entry in frames if entry['face_path'] and os.path.exists(os.path.join(root_dir, entry['face_path']))]
                body_paths = [entry['body_path'] for entry in frames if entry['body_path'] and os.path.exists(os.path.join(root_dir, entry['body_path']))]

            # Extract Landmarks (Face Normalized + Body Normalized)
            lm_face_outer_lips = [entry.get('mouth_outer_norm_face', []) for entry in frames]
            lm_face_inner_lips = [entry.get('mouth_inner_norm_face', []) for entry in frames] 
            lm_face_cheeks = [entry.get('cheek_norm_face', []) for entry in frames]
            
            lm_body_outer_lips = [entry.get('mouth_outer_norm_body', []) for entry in frames]
            lm_body_inner_lips = [entry.get('mouth_inner_norm_body', []) for entry in frames]
            lm_body_cheeks = [entry.get('cheek_norm_body', []) for entry in frames]
            
            lm_pose = [entry.get('pose_norm_body', []) for entry in frames]
            lm_l_hand = [entry.get('left_hand_norm_body', []) for entry in frames]
            lm_r_hand = [entry.get('right_hand_norm_body', []) for entry in frames]

            # Validation & Length Checks
            if self.mode == 'face' and len(face_paths) == 0: continue
            if self.mode == 'body' and len(body_paths) == 0: continue
            if self.mode == 'both' and (len(face_paths) == 0 or len(body_paths) == 0): continue
            
            if self.mode == 'no_img': seq_len = len(lm_pose)
            elif self.mode == 'face': seq_len = len(face_paths)
            elif self.mode == 'body': seq_len = len(body_paths)
            else: seq_len = min(len(face_paths), len(body_paths))
                
            if seq_len == 0: continue
            if (self.min_seq_len and seq_len < self.min_seq_len) or (self.max_seq_len and seq_len > self.max_seq_len):
                continue

            self.samples.append({
                'segment_key': segment_key + '#' + video_name,
                'face_paths': face_paths,
                'body_paths': body_paths,
                'label': label_str,
                'lm_face_outer_lips': lm_face_outer_lips,
                'lm_face_inner_lips': lm_face_inner_lips,
                'lm_face_cheeks': lm_face_cheeks,
                'lm_body_outer_lips': lm_body_outer_lips,
                'lm_body_inner_lips': lm_body_inner_lips,
                'lm_body_cheeks': lm_body_cheeks,
                'lm_pose': lm_pose,
                'lm_l_hand': lm_l_hand,
                'lm_r_hand': lm_r_hand,
                'length': seq_len,
                'video_path': self.root_dir 
            })

        # --- Set Label Mapping based on Task ---
        if label_mapping is not None:
            self.label_mapping = label_mapping
        else:
            if self.task == 'classification':
                self.label_mapping = {
                    'no-mouth-movement': 0, 'Mouthing': 1, 'MouthGesture': 2, 'Others': 3
                }
            elif self.task == 'localization':
                self.label_mapping = {
                    'no-mouth-movement': 0, 'movement': 1
                }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        clean_label = item['label']

        if clean_label not in self.label_mapping:
            raise ValueError(f"Unknown label: '{clean_label}' for task '{self.task}'. Expected one of {list(self.label_mapping.keys())}")
        
        sample = {
            'label': torch.tensor(self.label_mapping[clean_label], dtype=torch.long),
            'length': item['length']
        }

        # Load Face Images
        if self.mode in ['face', 'both']:
            face_seq = [self.face_transform(Image.open(os.path.join(item['video_path'], p)).convert('RGB')) for p in item['face_paths']]
            if face_seq: sample['face_seq'] = torch.stack(face_seq)

        # Load Body Images
        if self.mode in ['body', 'both']:
            body_seq = [self.body_transform(Image.open(os.path.join(item['video_path'], p)).convert('RGB')) for p in item['body_paths']]
            if body_seq: sample['body_seq'] = torch.stack(body_seq)

        # Load Landmarks
        if self.return_landmarks:
            sample['lm_face_outer_lips'] = torch.tensor(item['lm_face_outer_lips'], dtype=torch.float)
            sample['lm_face_inner_lips'] = torch.tensor(item['lm_face_inner_lips'], dtype=torch.float)
            if item['lm_face_cheeks'] and item['lm_face_cheeks'][0]:
                sample['lm_face_cheeks'] = torch.tensor(item['lm_face_cheeks'], dtype=torch.float)
                
            sample['lm_body_outer_lips'] = torch.tensor(item['lm_body_outer_lips'], dtype=torch.float)
            sample['lm_body_inner_lips'] = torch.tensor(item['lm_body_inner_lips'], dtype=torch.float)
            if item['lm_body_cheeks'] and item['lm_body_cheeks'][0]:
                sample['lm_body_cheeks'] = torch.tensor(item['lm_body_cheeks'], dtype=torch.float)
                
            sample['lm_pose'] = torch.tensor(item['lm_pose'], dtype=torch.float)
            sample['lm_l_hand'] = torch.tensor(item['lm_l_hand'], dtype=torch.float)
            sample['lm_r_hand'] = torch.tensor(item['lm_r_hand'], dtype=torch.float)

        return sample


class MultiVideoFaceBodySequenceDataset(Dataset):
    def __init__(self, root_dir, task='classification', face_transform=None, body_transform=None, label_mapping=None,
                 return_landmarks=True, mode='both', min_seq_len=None, max_seq_len=None, downsample=0, annotation_end_info=None):
        self.samples = []
        
        for subdir in sorted(os.listdir(root_dir)):
            sub_path = os.path.join(root_dir, subdir)
            if not os.path.isdir(sub_path): continue

            json_path = os.path.join(sub_path, f"{subdir}.json")
            if not os.path.isfile(json_path): continue

            try:
                video_dataset = FaceBodySequenceDataset(
                    json_path=json_path, root_dir=sub_path, task=task, face_transform=face_transform, body_transform=body_transform,
                    label_mapping=label_mapping, return_landmarks=return_landmarks, mode=mode,
                    min_seq_len=min_seq_len, max_seq_len=max_seq_len, annotation_end_info=annotation_end_info
                )
                self.samples.extend(video_dataset.samples)
            except Exception as e:
                print(f"[WARN] Skipping {json_path} due to error: {e}")

        self.label_mapping = video_dataset.label_mapping if len(self.samples) > 0 else {}
        self.face_transform = face_transform
        self.body_transform = body_transform
        self.return_landmarks = return_landmarks
        self.mode = mode

        # --- Unified Downsampling Logic ---
        if downsample > 0 and len(self.samples) > 0:
            label_groups = defaultdict(list)
            for s in self.samples: label_groups[s['label']].append(s)

            class_counts = {label: len(samples) for label, samples in label_groups.items()}
            sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

            balanced_samples = []
            if downsample == 1 and len(sorted_counts) >= 2:
                target_count = sorted_counts[1][1]
                largest_label = sorted_counts[0][0]
                for label, samples in label_groups.items():
                    balanced_samples.extend(random.sample(samples, target_count) if label == largest_label else samples)
            elif downsample == 2:
                target_count = min(class_counts.values())
                for label, samples in label_groups.items():
                    balanced_samples.extend(random.sample(samples, target_count))
            else:
                balanced_samples = self.samples
                
            self.samples = balanced_samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        dummy_dataset = FaceBodySequenceDataset.__new__(FaceBodySequenceDataset)
        dummy_dataset.label_mapping = self.label_mapping
        dummy_dataset.mode = self.mode
        dummy_dataset.face_transform = self.face_transform
        dummy_dataset.body_transform = self.body_transform
        dummy_dataset.return_landmarks = self.return_landmarks
        dummy_dataset.samples = self.samples
        dummy_dataset.task = "dummy" 
        return dummy_dataset[idx]


def collate_fn_face_body_sequence(batch):
    labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)
    lengths = torch.tensor([b['length'] for b in batch], dtype=torch.long)
    batch_out = {'labels': labels, 'lengths': lengths}

    if 'face_seq' in batch[0]:
        batch_out['face_seq'] = pad_sequence([b['face_seq'] for b in batch], batch_first=True)
    if 'body_seq' in batch[0]:
        batch_out['body_seq'] = pad_sequence([b['body_seq'] for b in batch], batch_first=True)

    # Pad all spatial landmarks
    for lm_key in ['lm_face_outer_lips', 'lm_face_inner_lips', 'lm_face_cheeks', 
                   'lm_body_outer_lips', 'lm_body_inner_lips', 'lm_body_cheeks', 
                   'lm_pose', 'lm_l_hand', 'lm_r_hand']:
        if lm_key in batch[0]:
            batch_out[lm_key] = pad_sequence([b[lm_key] for b in batch], batch_first=True)

    return batch_out

class BucketedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, sort_key='length'):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.sort_key = sort_key

        # Sort indices by length
        self.sorted_indices = sorted(range(len(dataset)), key=lambda i: dataset.samples[i][sort_key])
        # Bucket the sorted indices into batches
        self.batches = [
            self.sorted_indices[i:i + batch_size]
            for i in range(0, len(self.sorted_indices), batch_size)
        ]

        if self.drop_last and len(self.batches) > 0 and len(self.batches[-1]) < batch_size:
            self.batches = self.batches[:-1]

    def __iter__(self):
        if self.shuffle: random.shuffle(self.batches)
        for batch in self.batches: yield batch

    def __len__(self): return len(self.batches)