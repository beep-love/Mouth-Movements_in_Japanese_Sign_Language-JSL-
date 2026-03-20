# Sign Language Multimodal Data Pipeline

A complete end-to-end data engineering pipeline for Continuous Sign Language Recognition (CSLR). This pipeline ingests raw videos and multi-annotator ELAN files, resolves annotation conflicts, extracts normalized spatial landmarks via MediaPipe, and prepares dynamically padded PyTorch tensors for model training.

## Pipeline Architecture

The workflow is broken down into five sequential modules:

### 1. Annotation & Video Preprocessing (`eaf_preprocessor.py`)
Handles raw data standardization and ground truth generation.
* Converts `.mov` files to `.mp4` using FFmpeg.
* Fixes filename inconsistencies and typo corrections.
* Consolidates multiple ELAN annotators into a single Ground Truth (`.with_GT.eaf`) using majority consensus for classification and union blocks for localization.
* Generates Cohen's Kappa agreement metrics.
* **Usage:** `python eaf_preprocessor.py --input_dir <RAW_EAFS> --video_dir <RAW_VIDEOS> --out_dir <PROCESSED_DIR>`

### 2. Feature Extraction (`preprocess.py`)
The core computer vision engine powered by MediaPipe Holistic.
* Performs a fast, one-pass linear read of the videos.
* Extracts high-resolution Face crops and Body crops with padding.
* Computes bounding-box-normalized `[0, 1]` coordinates for the face mesh, pose, and hands wrt the face crops and body crops.
* Utilizes **Forward-Filling** to handle missing hand landmarks seamlessly.
* **Usage:** `python preprocess.py --video_folder <RAW_VIDEOS> --eaf_folder <PROCESSED_EAFS> --save_root <EXTRACTED_FEATURES> --extract_images`

### 3. Exploratory Data Analysis (`run_eda.py`)
Analyzes the extracted dataset to generate crucial training hyperparameters.
* Calculates dataset-wide pixel Mean and Standard Deviation for PyTorch normalization.
* Uses tensor math to accurately detect forward-filled (missing) hand frames.
* Generates distribution histograms and class balance reports.
* **Usage:** `python run_eda.py --root_dir <EXTRACTED_FEATURES> --base_save_dir <EDA_RESULTS> --downsample 1`

### 4. PyTorch Integration (`jsl_dataset.py`)
A robust data loader designed for neural network ingestion.
* Implements `MultiVideoFaceBodySequenceDataset` to recursively load JSON metadata.
* Applies independent transforms to the Face (112x112) and Body (224x224) crops.
* Features automatic class balancing (downsampling) and sequence length filtering.
* Uses `collate_fn_face_body_sequence` to dynamically zero-pad variable-length sequences into uniform training batches.
* **Usage:** Import directly into PyTorch training loop.

### 5. Presentation & Debugging (`generate_input_video.py`)
A visualization tool that recreates exactly what the neural network sees.
* Loads the processed PyTorch tensors and scales them back to visual space.
* Upscales the face crop dynamically using nearest-neighbor interpolation to prevent blurring.
* Draws bounding-box relative landmarks directly onto the crops.
* Stitches the Face and Body crops into a side-by-side `.mp4` video with frame-by-frame text overlays.
* **Usage:** `python generate_input_video.py --root_dir <EXTRACTED_FEATURES> --save_dir <OUTPUT_DIR> --fps 15`

## Dependencies
* `torch`, `torchvision`
* `opencv-python`
* `mediapipe`
* `pympi-ling`
* `matplotlib`
* `numpy`, `tqdm`
* System requirement: `ffmpeg` (for video conversion)