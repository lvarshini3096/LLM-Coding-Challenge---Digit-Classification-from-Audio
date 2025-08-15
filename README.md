# LLM-Coding-Challenge - Digit Classification from Audio
This project is an end-to-end audio classification system built to recognize spoken digits (0-9). It uses PyTorch to train a Convolutional Neural Network (CNN) on Mel-Frequency Cepstral Coefficients (MFCCs) extracted from the [Free Spoken Digit Dataset (FSDD)](https://github.com/Jakobovski/free-spoken-digit-dataset).

The project includes scripts for data preparation, model training, file-based prediction, and real-time inference from a microphone. 'cloudwalk_challenge.mp4' video uploaded to include the work progress. I have compressed the video to fit the 25mb limit, please let me know if you require a better quality video. 

## Features

- **Robust Data Pipeline:** Handles variable-length audio, data augmentation (noise, gain, time shift), and train/validation/test splitting.
- **CNN Model:** A simple but effective CNN architecture for classifying 2D MFCC feature maps.
- **File-Based Inference:** Scripts to predict digits from a directory of `.wav` files.
- **Real-Time Inference:** A script to capture live audio from a microphone and perform predictions.
- **Voice Activity Detection (VAD):** The real-time inference uses a simple energy-based VAD to find the most relevant part of the audio, improving accuracy.

---

## File Organization

The project is organized into several Python scripts, each with a specific purpose:

```
spoken-digit-classifier/
├── free-spoken-digit-dataset/  # Contains the FSDD audio files
│   ├── recordings/
│   └── inference_audio/
├── data_preparation.py         # Handles dataset loading, augmentation, and dataloaders.
├── model.py                    # Defines the CNNClassifier architecture.
├── train.py                    # Main script for training and evaluating the model.
├── predict_file.py             # Runs predictions on all audio files in a directory.
├── inference.py                # Runs real-time inference using the microphone.
├── check_mic.py                # A utility to diagnose microphone settings.
├── audio_feature_analyzer.py   # (Archived) Script for initial data exploration.
├── fsdd_cnn_model.pth          # The trained model weights (output of train.py).
└── requirements.txt            # Project dependencies.
```

- **`data_preparation.py`**: Contains the `FSDDDataset` class and `create_dataloaders` function. It is the core of the data pipeline, responsible for loading audio, applying augmentations, and preparing batches for the model.
- **`model.py`**: Defines the `CNNClassifier` `nn.Module`, which consists of convolutional layers for feature extraction and linear layers for classification.
- **`train.py`**: The main training script. It orchestrates the data loading, model instantiation, training loop, validation, and final testing. It saves the best-performing model weights to `fsdd_cnn_model.pth`.
- **`predict_file.py`**: A utility script to run batch predictions on all `.wav` files located in the `inference_audio` directory.
- **`free-spoken-digit-dataset/inference_audio/`**: This folder contains 10 audio files that were manually separated from the main dataset *before* any training or validation. These files serve as a true hold-out set for inference testing.
- **`inference.py`**: The real-time application. It listens to the microphone, performs a simple energy-based Voice Activity Detection (VAD), and predicts the spoken digit.
- **`check_mic.py`**: A small diagnostic tool to list available audio devices and verify the default microphone is configured correctly.

---

## Methodology and Approach

The project was developed in four distinct phases:

### 1. Feature Exploration
Before building a model, I used `librosa` and `matplotlib` to visually inspect the audio features. I extracted and studied the MFCCs for distinct patterns for different digits, making them a suitable input for a classifier. 

### 2. Data Pipeline Construction
Built a robust data pipeline in `data_preparation.py` with several key considerations:
- **Fixed-Length Input:** All audio clips were padded or truncated to a fixed 1-second length to enable batch processing.
- **Data Augmentation:** To improve model generalization, applied random augmentations (time shifting, gain changes, and adding Gaussian noise) exclusively to the training set.
- **Efficient Loading:** The pipeline uses a custom PyTorch `Dataset` and `DataLoader` to efficiently load and preprocess data in shuffled batches.

### 3. Model Training
With a solid data pipeline, we focused on training the model in `train.py`:
- **Architecture:** A `CNNClassifier` was designed in `model.py` to learn patterns from the 2D MFCC feature maps.
- **Training Loop:** A standard training loop was implemented, which calculates loss (`CrossEntropyLoss`), backpropagates gradients, and updates model weights using the `Adam` optimizer.
- **Validation & Best Model Saving:** After each training epoch, the model's performance was measured on a separate validation set. The model's state was only saved to `fsdd_cnn_model.pth` if its validation accuracy improved, ensuring that we keep the best version of the model and avoid overfitting.
- **Final Testing:** After the training loop completed, the best saved model was loaded and evaluated on the held-out test set to get an unbiased measure of its final performance.

### 4. Inference and Application
The final phase was to use the trained model for prediction:
- **File-Based Prediction:** `predict_file.py` was created for batch testing. Crucially, this was tested on the 10 manually separated audio files in the `inference_audio` folder. Since the model had never seen these files, this provided a reliable confirmation that the saved model was valid.
- **Real-Time Inference:** `inference.py` was developed to provide a live demonstration. After several iterations of debugging, the most robust solution was implemented:
    1.  Record a 1.5-second clip to provide a buffer.
    2.  Slide a 1-second window across the clip to find the segment with the highest audio energy (RMS).
    3.  Feed only this "most energetic chunk" to the model for prediction. This simple VAD logic was used to achieving reliable real-time performance.

---

## How to Use

### 1. Setup

First, clone the repository and install the required dependencies. It is recommended to use a virtual environment.

```bash
# Clone the repository (if you haven't already)
# git clone ...

# Navigate to the project directory
# cd spoken-digit-classifier

# Install dependencies
pip install -r requirements.txt

# Download the FSDD dataset and place it in the project root
# The 'recordings' folder should be at ./free-spoken-digit-dataset/recordings/
```

### 2. Training the Model

Run the `train.py` script to start the training process. This will train the model, evaluate it, and save the best version as `fsdd_cnn_model.pth`.

```bash
python train.py
```

### 3. Predicting from Audio Files

The `free-spoken-digit-dataset/inference_audio/` directory has been pre-populated with 10 files that were held out from the training, validation, and test sets. You can also add your own `.wav` files to this folder. Run the `predict_file.py` script to test them.

```bash
python predict_file.py
```
The script will automatically find and predict all `.wav` files in that folder.

### 4. Real-Time Prediction

To use your microphone for live prediction, simply run the `inference.py` script.

```bash
python inference.py
```
Follow the on-screen prompts to record your voice and see the prediction.

If you encounter issues, you can diagnose your microphone setup by running:
```bash
python check_mic.py
```
