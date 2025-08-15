import os
import torch
import torchaudio
import random
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class FSDDDataset(Dataset):
    """
    A PyTorch Dataset for the Free Spoken Digit Dataset (FSDD).
    """
    def __init__(self, annotations, transform, target_sample_rate, fixed_len_samples,
                 apply_augmentation=False):
        """
        Args:
            annotations (list): A list of tuples (audio_path, label).
            transform (callable): A transform to be applied on the fixed-length waveform.
            target_sample_rate (int): The sample rate to which all audio will be resampled.
            fixed_len_samples (int): The target length for all waveforms (in samples).
            apply_augmentation (bool): Whether to apply data augmentation.
        """
        self.annotations = annotations
        self.transform = transform
        self.target_sample_rate = target_sample_rate
        self.fixed_len_samples = fixed_len_samples
        self.apply_augmentation = apply_augmentation

        self.resampler = None

    def __len__(self):
        return len(self.annotations)

    def _fix_audio_length(self, waveform):
        """Pads or truncates a waveform to the fixed length."""
        current_length = waveform.shape[1]
        if current_length > self.fixed_len_samples:
            waveform = waveform[:, :self.fixed_len_samples]
        elif current_length < self.fixed_len_samples:
            padding_needed = self.fixed_len_samples - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding_needed))
        return waveform

    def _time_shift(self, waveform, shift_limit=0.1):
        """Applies a random time shift."""
        sig_len = waveform.shape[1]
        shift_amt = int(random.random() * shift_limit * sig_len)
        return torch.roll(waveform, shifts=shift_amt, dims=1)

    def _change_gain(self, waveform, min_gain_db=-10, max_gain_db=5):
        """Changes the gain (volume) of the waveform."""
        gain_db = random.uniform(min_gain_db, max_gain_db)
        gain_amplitude = 10**(gain_db / 20.0)
        return waveform * gain_amplitude

    def _add_background_noise(self, waveform, min_snr_db=0, max_snr_db=15):
        """Adds random Gaussian noise to the waveform."""
        # Generate noise with the same shape as the waveform
        noise = torch.randn_like(waveform)
        
        # Add noise at a random Signal-to-Noise Ratio (SNR)
        return torchaudio.functional.add_noise(waveform, noise, snr=torch.tensor([random.uniform(min_snr_db, max_snr_db)]))

    def __getitem__(self, index):
        audio_path, label = self.annotations[index]
        
        # 1. Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 2. Resample if necessary
        if sample_rate != self.target_sample_rate:
            if self.resampler is None:
                self.resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            waveform = self.resampler(waveform)
            
        # 3. Apply augmentations (if enabled) BEFORE padding
        if self.apply_augmentation:
            if random.random() > 0.5: waveform = self._time_shift(waveform)
            if random.random() > 0.5: waveform = self._change_gain(waveform)
            if random.random() > 0.5: waveform = self._add_background_noise(waveform)

        # 4. Fix length (pad/truncate)
        waveform = self._fix_audio_length(waveform)
        
        # 5. Apply MFCC transform
        features = self.transform(waveform)
        
        return features, torch.tensor(label, dtype=torch.long)

def create_dataloaders(data_dir, batch_size, split_ratios=(0.8, 0.1, 0.1)):
    """
    Creates training, validation, and test dataloaders for the FSDD dataset.
    The training set is doubled in size, containing one clean copy and one augmented
    copy of each original training sample.
    """
    # --- Configuration for Transformations ---
    SAMPLE_RATE = 8000
    FIXED_LEN_SECONDS = 1
    FIXED_LEN_SAMPLES = SAMPLE_RATE * FIXED_LEN_SECONDS

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=13,
        melkwargs={'n_fft': 256, 'hop_length': 128, 'n_mels': 23, 'center': True}
    )

    # --- Scan directory and create a list of all annotations ---
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Audio directory not found at: {data_dir}")
    
    all_annotations = []
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".wav"):
            try:
                label = int(filename.split('_')[0])
                all_annotations.append((os.path.join(data_dir, filename), label))
            except (ValueError, IndexError):
                print(f"Warning: Could not parse label from filename: {filename}")

    print(f"Total number of samples found: {len(all_annotations)}")
    random.shuffle(all_annotations) # Shuffle before splitting

    # --- Split annotations list ---
    num_samples = len(all_annotations)
    num_train = int(num_samples * split_ratios[0])
    num_val = int(num_samples * split_ratios[1]) # test set gets the remainder

    train_annotations = all_annotations[:num_train]
    val_annotations = all_annotations[num_train : num_train + num_val]
    test_annotations = all_annotations[num_train + num_val:]

    print(f"Original split: {len(train_annotations)} train, {len(val_annotations)} validation, {len(test_annotations)} test samples.")

    # --- Create separate Dataset instances ---
    # Create a clean version of the training set
    clean_train_dataset = FSDDDataset(train_annotations, mfcc_transform, SAMPLE_RATE, FIXED_LEN_SAMPLES,
                                      apply_augmentation=False)

    # Create an augmented version of the training set
    augmented_train_dataset = FSDDDataset(train_annotations, mfcc_transform, SAMPLE_RATE, FIXED_LEN_SAMPLES,
                                          apply_augmentation=True)

    # Combine them to double the training set size
    train_dataset = ConcatDataset([clean_train_dataset, augmented_train_dataset])
    print(f"Final training set size (with augmentations): {len(train_dataset)}")

    val_dataset = FSDDDataset(val_annotations, mfcc_transform, SAMPLE_RATE, FIXED_LEN_SAMPLES)
    test_dataset = FSDDDataset(test_annotations, mfcc_transform, SAMPLE_RATE, FIXED_LEN_SAMPLES)

    # --- Create DataLoaders ---
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
