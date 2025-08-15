import torch
import torchaudio
import argparse
from model import CNNClassifier
import sounddevice as sd
import numpy as np

# --- Configuration ---
SAMPLE_RATE = 8000
FIXED_LEN_SECONDS = 1
FIXED_LEN_SAMPLES = SAMPLE_RATE * FIXED_LEN_SECONDS
MODEL_PATH = "fsdd_cnn_model.pth"
NUM_CLASSES = 10
RECORDING_DURATION_S = 1.5 # Record for longer to capture the full word
RECORDING_SAMPLES = int(SAMPLE_RATE * RECORDING_DURATION_S)

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
    parser = argparse.ArgumentParser(description="Predict a spoken digit. Runs in live mode by default.")
    parser.add_argument("audio_file", type=str, nargs='?', default=None, help="Optional. Path to a .wav file to run in file mode.")
    args = parser.parse_args()

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

    # 3. Get audio waveform either from file or live recording
    if args.audio_file is None: # Live Mode
        while True:
            user_input = input("\nPress Enter to record for 1 second (or 'q' to quit): ")
            if user_input.lower() == 'q':
                break

            # Record audio for a longer duration to not miss the word
            print(f"Recording for {RECORDING_DURATION_S} seconds...")
            recording = sd.rec(RECORDING_SAMPLES, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            print("...finished recording.")

            # Find the most energetic 1-second chunk in the recording.
            # This acts as a simple voice activity detection (VAD).
            num_windows = RECORDING_SAMPLES - FIXED_LEN_SAMPLES
            if num_windows > 0:
                max_rms = 0
                best_window_start = 0
                for i in range(num_windows):
                    window = recording[i : i + FIXED_LEN_SAMPLES]
                    rms = np.sqrt(np.mean(window**2))
                    if rms > max_rms:
                        max_rms = rms
                        best_window_start = i
                
                # Extract the most energetic 1-second chunk
                best_chunk = recording[best_window_start : best_window_start + FIXED_LEN_SAMPLES]
            else:
                best_chunk = recording

            # Convert the best chunk to a torch tensor with a channel dimension
            waveform = torch.from_numpy(best_chunk).T
            
            # Apply MFCC transform and predict
            mfcc_features = mfcc_transform(waveform)
            predicted_digit = predict(model, mfcc_features, device)

            print(f"--> The model predicts the digit: {predicted_digit}")
    else: # File Mode
        print(f"\n--- File Mode: Processing '{args.audio_file}' ---")
        try:
            waveform, sr = torchaudio.load(args.audio_file)
        except FileNotFoundError:
            print(f"Error: Audio file not found at '{args.audio_file}'")
            return
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return

        # Resample to 8kHz if necessary
        if sr != SAMPLE_RATE:
            print(f"Resampling audio from {sr}Hz to {SAMPLE_RATE}Hz...")
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad or truncate to the fixed length used during training
        if waveform.shape[1] < FIXED_LEN_SAMPLES:
            waveform = torch.nn.functional.pad(waveform, (0, FIXED_LEN_SAMPLES - waveform.shape[1]))
        waveform = waveform[:, :FIXED_LEN_SAMPLES]

        # Apply MFCC transform
        mfcc_features = mfcc_transform(waveform)

        # 4. Make a prediction
        predicted_digit = predict(model, mfcc_features, device)

        print("\n--- Prediction ---")
        print(f"The model predicts the digit: {predicted_digit}")

if __name__ == "__main__":
    main()
