#############################################################################
###  adopted from Tim Henrik Sandermann implementation at NII internship  ###
#############################################################################

import json
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sympy import Polygon, Point2D
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import Counter, defaultdict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import zscore
import pandas as pd
import seaborn as sns
import random
from matplotlib.cm import get_cmap
import pickle

from src.data.jsl_dataset import MultiVideoFaceMouthSequenceDataset
from src.models.localization.utils import MouthMovementDetector, StandardMouthMovementDetector

MOUTH_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                       291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
MOUTH_INNER_LANDMARKS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
                             308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

CLASS_ENCODER_FOUR={
    'no-mouth-movement': 0,
    'movement': 1,
    'no-mouth-movement*movement': 2,
    'movement*no-mouth-movement': 3,
}
CLASS_ENCODER_TWO={
    'no-mouth-movement': 0,
    'movement': 0,
    'no-mouth-movement*movement': 1,
    'movement*no-mouth-movement': 1,
}
CLASS_ENCODER_THREE={
    'no-mouth-movement': 0,
    'movement': 1,
    'no-mouth-movement*movement': 2,
    'movement*no-mouth-movement': 2,
}

CLASS_ENCODER_SIMPLE = {
    'no-mouth-movement': 0,
    'movement': 1,
    'no-mouth-movement*movement': -1,
    'movement*no-mouth-movement': -1,
}

FEATURE_NAMES = [
        "mean_area",
        "mean_width", "max_diff_width",
        "max_diff_height",
        "max_spike_area"
        #"max_spike_height", "mean_height", "max_diff_area",
    ]

# First entry for outer and second for inner lip
UPPER_LIP = [0, 13]
LOWER_LIP = [17, 14]
LEFT_LIP = [61, 78]
RIGHT_LIP = [291, 308]     # may check ths again


class RFMouthLandmarkDetector(StandardMouthMovementDetector):
    def __init__(self, model_path=None):
        super().__init__(model_path)

    def create_model(self):
        self.model = RandomForestClassifier()

class LRMouthLandmarkDetector(StandardMouthMovementDetector):
    def __init__(self, model_path=None):
        super().__init__(model_path)

    def create_model(self):
        self.model = LogisticRegression(multi_class='auto')


# Helpers
def map_coords_to_landmarks(coords, inner_mouth, landmarks=None):
    if landmarks is None:
        landmarks = MOUTH_LANDMARKS if not inner_mouth else MOUTH_INNER_LANDMARKS

    assert len(coords) == len(landmarks)
    return dict(zip(landmarks, np.array(coords)))


def moving_average(series, window_size=3):
    return np.convolve(series, np.ones(window_size)/window_size, mode='same')


def normalize_window_length(data_samples, sequence_keys=['landmarks_mouth'], target_length=10):
    print("[INFO] Normalizing sequences to target length:", target_length)

    def normalize_sequences(sequence_lists):
        # Convert all to arrays and ensure T is consistent
        sequence_arrays = [np.array(seq) for seq in sequence_lists]  # Each: (T, L, D)
        T = sequence_arrays[0].shape[0]

        if T == target_length:
            return [seq.tolist() for seq in sequence_arrays]

        x_old = np.linspace(0, 1, T)
        x_new = np.linspace(0, 1, target_length)

        normalized = []
        for seq in sequence_arrays:
            T_, L, D = seq.shape
            interpolated = np.zeros((target_length, L, D))
            for l in range(L):
                for d in range(D):
                    f = interp1d(x_old, seq[:, l, d], kind='linear')
                    interpolated[:, l, d] = f(x_new)
            normalized.append(interpolated.tolist())
        return normalized

    for sample in data_samples:
        # Extract lists for each sequence key
        sequence_lists = [sample[key] for key in sequence_keys]
        # Normalize them together
        normalized_sequences = normalize_sequences(sequence_lists)
        # Write them back
        for key, norm_seq in zip(sequence_keys, normalized_sequences):
            sample[key] = norm_seq

    return data_samples


# Intra-landmark features
def get_mouth_vertical_opening(mouth_dict, inner_mouth=True):
    return np.linalg.norm(mouth_dict[UPPER_LIP[int(inner_mouth)]] - mouth_dict[LOWER_LIP[int(inner_mouth)]])


def get_mouth_horizontal_length(mouth_dict, inner_mouth=True):
    return np.linalg.norm(mouth_dict[LEFT_LIP[int(inner_mouth)]] - mouth_dict[RIGHT_LIP[int(inner_mouth)]])


def get_mouth_area(mouth_dict, inner_mouth=True):
    landmarks = MOUTH_INNER_LANDMARKS if inner_mouth else MOUTH_LANDMARKS
    coords = np.array([mouth_dict[i] for i in landmarks])
    x = coords[:, 0]
    y = coords[:, 1]

    # Shoelace formula
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    return area


# Feature vector
def extract_feature_vector(area, width, height):
    # max jump between consecutive frames
    def spike_feature(values):
        return max(abs(values[i + 1] - values[i]) for i in range(len(values) - 1))

    return np.array([
        np.mean(area),
        np.mean(width), np.ptp(width),
        np.ptp(height),
        spike_feature(area),

        # Features with high correlations
        # spike_feature(height)
        # np.mean(height)
        # np.ptp(area)
        #
    ])


def prepare_data(mouth_dict_list, inner_mouth, num_classes=3):
    x, y = [], []
    x_names = []
    for mouth_dict in mouth_dict_list:
        if 'landmarks_mouth' not in mouth_dict:
            print(f"[WARN] Missing landmarks for sample: {mouth_dict}")
            continue

        label = mouth_dict['label']
        if num_classes == 2:
            label_encoding = CLASS_ENCODER_TWO[label]
        elif num_classes == 3:
            label_encoding = CLASS_ENCODER_THREE[label]
        elif num_classes == 4:
            label_encoding = CLASS_ENCODER_FOUR[label]
        elif num_classes == -1:
            label_encoding = CLASS_ENCODER_SIMPLE[label]
            if label_encoding == -1:
                continue
        else:
            raise ValueError(f"[WARN] Invalid number of classes: {num_classes}")

        y.append(label_encoding)

        area, height, width = [], [], []
        landmarks = mouth_dict['landmarks_mouth'] if not inner_mouth else mouth_dict['landmarks_mouth_inner']
        for raw_landmarks in landmarks:
            mouth_landmarks = map_coords_to_landmarks(raw_landmarks, inner_mouth)

            height.append(get_mouth_vertical_opening(mouth_landmarks, inner_mouth))
            width.append(get_mouth_horizontal_length(mouth_landmarks, inner_mouth))
            area.append(get_mouth_area(mouth_landmarks, inner_mouth))

        feature = extract_feature_vector(area, width, height)
        x.append(feature)
        x_names.append(mouth_dict['segment_key'])

    X = np.array(x)
    Y = np.array(y)
    return X, Y, x_names


def remove_outliers_per_class(X, Y, z_threshold=3):
    unique_classes = np.unique(Y)
    keep_mask = np.ones(len(X), dtype=bool)
    outlier_indices = []

    for cls in unique_classes:
        class_mask = (Y == cls)
        X_class = X[class_mask]

        # Compute Z-scores only for this class
        z = np.abs(zscore(X_class, axis=0))
        class_outlier_mask = np.any(z > z_threshold, axis=1)

        # Translate local indices to global
        global_indices = np.where(class_mask)[0][class_outlier_mask]
        outlier_indices.extend(global_indices)

        # Mark as to remove
        keep_mask[global_indices] = False

    print(f"[INFO] Removed {len(outlier_indices)} outliers (threshold = {z_threshold})")

    return X[keep_mask], Y[keep_mask], np.array(outlier_indices)


def analyze_prepared_data(mouth_dict_list, inner_mouth, num_classes, boxplots=False):
    X, Y = prepare_data(mouth_dict_list, inner_mouth, num_classes)

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df['Label'] = Y

    # Create box plots for each feature
    if boxplots:
        for feature in FEATURE_NAMES:
            plt.figure()
            df.boxplot(column=feature, by='Label')
            plt.title(f'{feature} by Label')
            plt.suptitle('')
            plt.xlabel('Label')
            plt.ylabel(feature)
            plt.grid(False)
            plt.tight_layout()
            plt.show()

    sns.pairplot(df, hue='Label', palette="deep")  # Show scatter plots of feature pairs
    plt.show()

    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()


# Simple supervised
def logistic_regression(mouth_dict_list, inner_mouth, num_classes=3, scale=True):
    X, Y = prepare_data(mouth_dict_list, inner_mouth, num_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    if scale:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model = LogisticRegression(multi_class='multinomial' if num_classes > 2 else 'auto')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", model.score(X_test, y_test))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()


def random_forest(mouth_dict_list, inner_mouth, num_classes=3, scale=True):
    X, Y, X_idx = prepare_data(mouth_dict_list, inner_mouth, num_classes)

    #X, Y, _ = remove_outliers_per_class(X, Y)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, Y, X_idx, test_size=0.2, random_state=42, stratify=Y
    )
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()

    print("Accuracy:", clf.score(X_test, y_test))

    importances = clf.feature_importances_
    feat_importance = pd.Series(importances, FEATURE_NAMES).sort_values(ascending=False)
    print(feat_importance)

    # wrong_mask = y_pred != y_test

    # print("Misclassified:", np.array(idx_test)[wrong_mask])

def svm(mouth_dict_list, inner_mouth, num_classes):
    X, Y = prepare_data(mouth_dict_list, inner_mouth, num_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    # 5. ROC-AUC (for binary classification)
    if len(set(Y)) == 2:
        auc = roc_auc_score(y_test, y_proba)
        print(f"\nROC AUC Score: {auc:.4f}")

# Clustering
def clustering_k_means(mouth_dict_list, k=4):
    two_classes = True if k == 2 else False
    X, Y = prepare_data(mouth_dict_list, two_classes)

    # Clustering
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(X)

    # Extract information (need conversion for json)
    summary = analyze_clusters([int(y) for y in Y], [int(l) for l in labels])

    with open("../cluster_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def analyze_clusters(y_true, cluster_labels):
    clusters = defaultdict(list)

    # Group true labels by predicted cluster
    for y, cluster in zip(y_true, cluster_labels):
        clusters[cluster].append(y)

    summary = []

    for cluster_id, labels in clusters.items():
        total = len(labels)
        label_counts = Counter(labels)
        dominant_label, count = label_counts.most_common(1)[0]
        purity = count / total

        summary.append({
            "cluster": int(cluster_id),
            "size": int(total),
            "dominant_label": int(dominant_label),
            "dominant_count": int(count),
            "purity": float(round(purity, 3)),
            "label_distribution": dict(label_counts)
        })

    return summary


# Feature Evaluation
def compare_landmark_features(mouth_dict_list):
    feature_values = {'vertical': {},  'horizontal': {}, 'area': {}}
    for mouth_dict in tqdm(mouth_dict_list):
        label = mouth_dict['label']
        if mouth_dict['label'] not in feature_values['vertical']:
            for feature in feature_values:
                feature_values[feature][label] = []

        for raw_landmarks in mouth_dict['landmarks_mouth']:
            mouth_landmarks = map_coords_to_landmarks(raw_landmarks)

            feature_values['vertical'][label].append(get_mouth_vertical_opening(mouth_landmarks))
            feature_values['horizontal'][label].append(get_mouth_horizontal_length(mouth_landmarks))
            feature_values['area'][label].append(get_mouth_area(mouth_landmarks))

    for feature in feature_values:
        plt.figure(figsize=(8, 6))
        data = [feature_values[feature][label] for label in feature_values[feature]]
        labels = list(feature_values[feature].keys())

        plt.boxplot(data, labels=labels)
        plt.title(f'{feature.capitalize()} by Label')
        plt.xlabel('Label')
        plt.ylabel(f'{feature.capitalize()} Value')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def compare_landmark_features_timescale(mouth_dict_list, smoothing=True, window_size=3, inner_mouth=True, mode=1):
    time_series_values = {'vertical': {}, 'horizontal': {}, 'area': {}}

    for mouth_dict in tqdm(mouth_dict_list):
        label = mouth_dict['label']

        for feature in time_series_values:
            if label not in time_series_values[feature]:
                time_series_values[feature][label] = []

        # Initialize time series for this sample
        vertical_series = []
        horizontal_series = []
        area_series = []

        mouth_landmarks = mouth_dict['landmarks_mouth'] if not inner_mouth else mouth_dict['landmarks_mouth_inner']
        for raw_landmarks in mouth_landmarks:
            mouth_landmarks = map_coords_to_landmarks(raw_landmarks, inner_mouth)

            vertical_series.append(get_mouth_vertical_opening(mouth_landmarks, inner_mouth))
            horizontal_series.append(get_mouth_horizontal_length(mouth_landmarks, inner_mouth))
            area_series.append(get_mouth_area(mouth_landmarks, inner_mouth))

        # Save time series per sample
        time_series_values['vertical'][label].append(vertical_series)
        time_series_values['horizontal'][label].append(horizontal_series)
        time_series_values['area'][label].append(area_series)


    if mode == 1:
        for feature in time_series_values:
            plt.figure(figsize=(10, 5))
            for label, sequences in time_series_values[feature].items():
                # Pad sequences to same length (optional but useful for plotting)
                max_len = max(len(seq) for seq in sequences)
                padded = np.array([
                    np.pad(seq, (0, max_len - len(seq)), constant_values=np.nan)
                    for seq in sequences
                ])
                mean = np.nanmean(padded, axis=0)
                # std = np.nanstd(padded, axis=0)
                perc_up = np.nanpercentile(padded, 90, axis=0)
                perc_low = np.nanpercentile(padded, 10, axis=0)

                if smoothing:
                    mean = moving_average(mean, window_size=window_size)
                    # std = moving_average(std, window_size=window_size)

                x = np.arange(len(mean))

                plt.plot(x, mean, label=f"{label} (mean)")
                # plt.fill_between(x, mean - std, mean + std, alpha=0.3)
                plt.fill_between(x, perc_low, perc_up, alpha=0.3)

            plt.title(f"{feature.capitalize()} over Time")
            plt.xlabel("Time Step")
            plt.ylabel(f"{feature.capitalize()} Value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    elif mode == 2:
        n_examples = 3

        # Color map for label colors
        label_names = list(time_series_values.keys())
        cmap = get_cmap("tab10")  # or "Set2", "Dark2", etc.
        label_colors = {label: cmap(i) for i, label in enumerate(time_series_values[label_names[0]].keys())}

        # Loop over features
        for feature in time_series_values:
            plt.figure(figsize=(10, 6))
            plt.title(f"Single Sequences per Label — Feature: {feature}")
            plt.xlabel("Timestep")
            plt.ylabel(feature)

            # Loop over labels
            for label, sequences in time_series_values[feature].items():
                k = min(n_examples, len(sequences))
                example_seqs = random.sample(sequences, k=k)
                for seq in example_seqs:
                    plt.plot(seq, color=label_colors[label], alpha=0.5, label=label)

            # Avoid duplicate legend entries
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            plt.tight_layout()
            plt.show()

def analyze_outliers(mouth_dict_list):
    X, Y = prepare_data(mouth_dict_list, two_classes=False)

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["label"] = Y

    n_features = len(FEATURE_NAMES)
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, feature in enumerate(FEATURE_NAMES):
        ax = axes[i]
        sns.histplot(data=df, x=feature, hue="label", kde=True, stat="density", common_norm=False, palette="Set2",
                     ax=ax)
        ax.set_title(f"{feature}")
        ax.grid(True)

    # Hide unused subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    num_classes = -1

    # Code 2 samples every class (of the 4 classes) to the same amount of samples -> unbalanced for 3 classes
    downsample_code = 1

    data = MultiVideoFaceMouthSequenceDataset(
        root_dir="/processed_annotations_0.3",
        mode='mouth',
        return_landmarks=True,
        transform=None,
        min_seq_len=11,
        max_seq_len=30,
        downsample=downsample_code)

    # compare_landmark_features(data.samples)

    #analyze_prepared_data(norm_data_samples, inner_mouth=True, num_classes=num_classes)
    #analyze_outliers(norm_data_samples)

    #compare_landmark_features_timescale(norm_data_samples, smoothing=False, window_size=1, inner_mouth=True, mode=1)
    #compare_landmark_features(data.samples)

    # clustering_k_means(data.samples, k=4)
    #logistic_regression(norm_data_samples, inner_mouth=True, num_classes=num_classes)
    random_forest(norm_data_samples, inner_mouth=True, num_classes=num_classes, scale=False)
    #svm(norm_data_samples, inner_mouth=True, num_classes=num_classes)

    """
    sample = data.samples[100]
    sample_coords = map_coords_to_landmarks(sample['landmarks_mouth'][0])

    print("Vertical mouth opening:", get_mouth_vertical_opening(sample_coords))
    print("Horizontal mouth opening:", get_mouth_horizontal_length(sample_coords))
    print("Area of mouth opening:", get_mouth_area(sample_coords))
    """
    # TODO: Calculate this per class and see if one notices something
    # TODO: or could calculate this for the 18 frames per case and see how it develops