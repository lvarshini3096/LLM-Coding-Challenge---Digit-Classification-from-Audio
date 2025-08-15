import torch
import torchaudio
from model import CNNClassifier
import os

# --- Configuration ---
SAMPLE_RATE = 8000
FIXED_LEN_SECONDS = 1
FIXED_LEN_SAMPLES = SAMPLE_RATE * FIXED_LEN_SECONDS
MODEL_PATH = "fsdd_cnn_model.pth"
NUM_CLASSES = 10
INFERENCE_AUDIO_DIR = '/cloudwalk_challenge/free-spoken-digit-dataset/inference_audio' #Folder containing audio files for inference

def predict(model, tensor, device):
    """
    Performs a forward pass on the model and returns the predicted class index.
    """
    model.eval()
    with torch.no_grad():
        tensor = tensor.to(device)
        # Add a batch dimension (B, C, H, W) as the model expects it
        tensor = tensor.unsqueeze(0)
        output = model(tensor)
        # Get the index of the max log-probability
        pred = output.argmax(dim=1, keepdim=True)
    return pred.item()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device for inference.")

    # 1. Load the trained model
    try:
        model = CNNClassifier(num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        print(f"Model loaded successfully from '{MODEL_PATH}'.")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'. Please run train.py first.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Define the same transformation pipeline used in training
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=13,
        melkwargs={'n_fft': 256, 'hop_length': 128, 'n_mels': 23, 'center': True}
    )

    # 4. Find all .wav files in the inference directory
    try:
        all_filenames = [f for f in os.listdir(INFERENCE_AUDIO_DIR) if f.endswith('.wav')]
        if not all_filenames:
            print(f"--> Error: No .wav files found in '{INFERENCE_AUDIO_DIR}'.")
            return
    except FileNotFoundError:
        print(f"--> Error: Inference directory not found at '{INFERENCE_AUDIO_DIR}'.")
        return

    print(f"--- Predicting all {len(all_filenames)} files in '{os.path.basename(INFERENCE_AUDIO_DIR)}' ---\n")

    # 5. Loop through each file, process, and predict
    for filename in sorted(all_filenames):
        audio_path = os.path.join(INFERENCE_AUDIO_DIR, filename)
        
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"File: {filename}")
            print(f"--> Error loading audio file: {e}\n")
            continue # Skip to the next file

        # Resample to 8kHz if necessary
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        # Convert to mono and fix length
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if waveform.shape[1] < FIXED_LEN_SAMPLES:
            waveform = torch.nn.functional.pad(waveform, (0, FIXED_LEN_SAMPLES - waveform.shape[1]))
        waveform = waveform[:, :FIXED_LEN_SAMPLES]

        # Apply MFCC transform and make prediction
        mfcc_features = mfcc_transform(waveform)
        predicted_digit = predict(model, mfcc_features, device)

        print(f"File: {filename}")
        print(f"--> Predicted: {predicted_digit}\n")

if __name__ == "__main__":
    main()
