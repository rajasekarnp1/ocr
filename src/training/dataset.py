# src/training/dataset.py
"""
PyTorch Dataset and DataLoader implementations for loading and preparing audio data.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
# from src.audio_io import read_audio # Use your project's audio reading
# from src.preprocessing import normalize_audio, get_mel_spectrogram # Use your project's preprocessing

# Fallback if src.audio_io or src.preprocessing are not found (e.g. running standalone)
try:
    from src.audio_io import read_audio
except ImportError:
    print("Warning: src.audio_io not found, using torchaudio for reading as fallback in dataset.py.")
    read_audio = None # Will be handled in read_audio_torchaudio_fallback

try:
    from src.preprocessing import normalize_audio, get_mel_spectrogram
except ImportError:
    print("Warning: src.preprocessing not found, dummy preprocessing will be used in dataset.py.")
    def normalize_audio(audio, target_db=-20.0): return audio # Dummy
    def get_mel_spectrogram(audio, sr, **kwargs): return torch.randn(128, 256) # Dummy, returns fixed size tensor


def read_audio_torchaudio_fallback(file_path, target_sr=None):
    """Fallback audio reading using torchaudio if project's read_audio is not available."""
    try:
        waveform, sr = torchaudio.load(file_path)
        if target_sr is not None and sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
            sr = target_sr
        if waveform.shape[0] > 1: # Convert to mono if stereo
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform.squeeze().numpy(), sr # Return as numpy array to match typical read_audio
    except Exception as e:
        print(f"Error reading audio file {file_path} with torchaudio: {e}")
        return None, None


class AudioFileDataset(Dataset):
    def __init__(self, audio_dir, target_sr=44100, segment_length_samples=None,
                 preprocessing_fn=None, file_extension='wav',
                 use_mel_spectrogram=False, mel_params=None):
        """
        Dataset for loading audio files from a directory.

        Args:
            audio_dir (str): Directory containing audio files.
            target_sr (int): Target sampling rate to resample audio to.
            segment_length_samples (int, optional): If provided, audio is randomly segmented.
                                                   Otherwise, full audio files are returned.
            preprocessing_fn (callable, optional): A function to apply to the loaded audio data.
                                                   Takes (audio_data, sr) as input.
            file_extension (str): Extension of audio files to look for (e.g., 'wav', 'mp3').
            use_mel_spectrogram (bool): If True, converts audio to mel spectrogram.
            mel_params (dict, optional): Parameters for mel spectrogram computation
                                         (e.g., n_fft, hop_length, n_mels).
        """
        self.audio_dir = audio_dir
        self.target_sr = target_sr
        self.segment_length_samples = segment_length_samples
        self.preprocessing_fn = preprocessing_fn
        self.use_mel_spectrogram = use_mel_spectrogram
        self.mel_params = mel_params if mel_params is not None else {}

        self.file_paths = []
        for root, _, files in os.walk(audio_dir):
            for file in files:
                if file.lower().endswith(f".{file_extension.lower()}"):
                    self.file_paths.append(os.path.join(root, file))

        if not self.file_paths:
            print(f"Warning: No audio files with extension '{file_extension}' found in {audio_dir}.")

        self._read_audio_fn = read_audio if read_audio is not None else read_audio_torchaudio_fallback
        print(f"AudioFileDataset initialized with {len(self.file_paths)} files.")
        if self._read_audio_fn == read_audio_torchaudio_fallback:
            print("  Using torchaudio for audio reading.")
        else:
            print("  Using project's audio_io.read_audio for audio reading.")


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        try:
            # audio_data is numpy array, sr is int
            audio_data, sr = self._read_audio_fn(file_path, target_sr=self.target_sr)
            if audio_data is None: # Handle read error
                print(f"Warning: Could not read {file_path}, returning zeros.")
                # Return a dummy tensor of expected type/shape if possible, or raise error
                # For now, return zeros, but this might cause issues downstream
                dummy_len = self.segment_length_samples if self.segment_length_samples else self.target_sr # 1 sec dummy
                if self.use_mel_spectrogram:
                    return torch.zeros((self.mel_params.get('n_mels', 128), dummy_len // self.mel_params.get('hop_length', 512))), idx
                else:
                    return torch.zeros(dummy_len), idx


            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_data).float()

            # Apply custom preprocessing if provided
            if self.preprocessing_fn:
                audio_tensor, sr = self.preprocessing_fn(audio_tensor, sr) # Assuming fn handles tensor or numpy

            # Normalize audio (example, can be part of preprocessing_fn)
            # Assuming normalize_audio expects numpy and returns numpy
            # If it works with tensors, this can be simplified.
            # For now, let's assume it's handled by preprocessing_fn or not done here.
            # audio_data_normalized = normalize_audio(audio_data, target_db=-20.0)
            # audio_tensor = torch.from_numpy(audio_data_normalized).float()


            # Random segmentation
            if self.segment_length_samples and len(audio_tensor) > self.segment_length_samples:
                start = torch.randint(0, len(audio_tensor) - self.segment_length_samples + 1, (1,)).item()
                audio_tensor = audio_tensor[start : start + self.segment_length_samples]
            elif self.segment_length_samples and len(audio_tensor) < self.segment_length_samples:
                # Pad if shorter than segment length
                padding = self.segment_length_samples - len(audio_tensor)
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
            elif self.segment_length_samples and len(audio_tensor) == self.segment_length_samples:
                pass # Length is already correct


            if self.use_mel_spectrogram:
                # Ensure audio_tensor is suitable for get_mel_spectrogram (e.g. numpy if librosa based)
                # If get_mel_spectrogram is from src.preprocessing, it should handle tensor or numpy
                # This might need adjustment based on the actual implementation of get_mel_spectrogram
                # For this example, assuming get_mel_spectrogram takes numpy array:
                mel_spec = get_mel_spectrogram(audio_tensor.numpy(), sr, **self.mel_params)
                output = torch.from_numpy(mel_spec).float()
            else:
                output = audio_tensor

            return output, idx # Return idx to potentially fetch metadata later if needed

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            # Fallback: return dummy data
            dummy_len = self.segment_length_samples if self.segment_length_samples else self.target_sr
            if self.use_mel_spectrogram:
                n_mels_dummy = self.mel_params.get('n_mels', 128)
                hop_dummy = self.mel_params.get('hop_length', 512)
                return torch.zeros((n_mels_dummy, dummy_len // hop_dummy if hop_dummy > 0 else 128)), idx
            else:
                return torch.zeros(dummy_len), idx


def collate_fn_pad(batch):
    """
    Collate function for DataLoader that pads sequences in a batch to the same length.
    Assumes batch is a list of (tensor_data, other_info_like_idx).
    """
    # Separate data and other info
    data_list = [item[0] for item in batch]
    other_info_list = [item[1] for item in batch] # e.g., indices

    # Pad data (assuming it's the first element of the tuple)
    # Works for both waveforms (2D: batch, time) and spectrograms (3D: batch, mels, time)
    are_spectrograms = data_list[0].ndim == 2 # (mels, time) before batching
    if are_spectrograms: # (batch, mels, time)
        # padding expects (...,time, mels) so transpose, pad, then transpose back
        # Or pad manually if features (mels) are fixed and only time varies.
        # For simplicity, let's assume feature dim is fixed and pad time.
        max_len = max(s.shape[1] for s in data_list)
        padded_data = []
        for s in data_list:
            pad_width = (0, max_len - s.shape[1]) # Pad only the time dimension (last dim)
            padded_s = torch.nn.functional.pad(s, pad_width, "constant", 0)
            padded_data.append(padded_s)
    else: # Waveforms (batch, time)
        max_len = max(s.shape[0] for s in data_list)
        padded_data = []
        for s in data_list:
            pad_width = (0, max_len - s.shape[0]) # Pad the time dimension
            padded_s = torch.nn.functional.pad(s, pad_width, "constant", 0)
            padded_data.append(padded_s)

    # Stack padded data
    data_tensor = torch.stack(padded_data)

    # Return padded data and other info (as a tensor if appropriate, or list)
    return data_tensor, torch.tensor(other_info_list)


def create_dataloader(audio_dir, batch_size, target_sr,
                      segment_length_samples=None,
                      preprocessing_fn=None,
                      use_mel_spectrogram=False, mel_params=None,
                      num_workers=0, shuffle=True, pin_memory=True,
                      file_extension='wav'):
    """
    Utility function to create a DataLoader for audio data.
    """
    dataset = AudioFileDataset(
        audio_dir=audio_dir,
        target_sr=target_sr,
        segment_length_samples=segment_length_samples,
        preprocessing_fn=preprocessing_fn,
        file_extension=file_extension,
        use_mel_spectrogram=use_mel_spectrogram,
        mel_params=mel_params
    )

    # If not using fixed segment_length_samples, full audio files are loaded,
    # and they might have variable lengths. In this case, collate_fn_pad is useful.
    # If segment_length_samples IS used, all tensors should be the same size already.
    current_collate_fn = collate_fn_pad if segment_length_samples is None else None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=current_collate_fn
    )
    return dataloader

if __name__ == '__main__':
    print("--- Audio Dataset and DataLoader Example ---")

    # Create a dummy audio directory with a few files for testing
    DUMMY_AUDIO_DIR = "dummy_audio_data_for_dataset_test"
    os.makedirs(DUMMY_AUDIO_DIR, exist_ok=True)

    SAMPLE_RATE = 16000
    # Create some dummy wav files using torchaudio
    try:
        for i in range(3):
            dummy_waveform = torch.sin(torch.arange(0, SAMPLE_RATE * (i + 1) * 0.5) * 0.1 * (i+1)).unsqueeze(0) # Varying lengths
            torchaudio.save(os.path.join(DUMMY_AUDIO_DIR, f"dummy_{i}.wav"), dummy_waveform, SAMPLE_RATE)

        # 1. Test AudioFileDataset (loading full files, variable length)
        print("\n1. Testing AudioFileDataset (full files):")
        full_file_dataset = AudioFileDataset(DUMMY_AUDIO_DIR, target_sr=SAMPLE_RATE)
        if len(full_file_dataset) > 0:
            sample_data, sample_idx = full_file_dataset[0]
            print(f"Sample data shape (full file): {sample_data.shape}, Type: {type(sample_data)}, Index: {sample_idx}")
        else:
            print("Full file dataset is empty (dummy files might not have been created).")

        # 2. Test AudioFileDataset (with segmentation)
        print("\n2. Testing AudioFileDataset (with segmentation):")
        SEGMENT_SAMPLES = SAMPLE_RATE // 2 # 0.5 seconds
        segmented_dataset = AudioFileDataset(DUMMY_AUDIO_DIR, target_sr=SAMPLE_RATE, segment_length_samples=SEGMENT_SAMPLES)
        if len(segmented_dataset) > 0:
            sample_data_seg, _ = segmented_dataset[0]
            print(f"Sample data shape (segmented): {sample_data_seg.shape}")
            assert sample_data_seg.shape[0] == SEGMENT_SAMPLES, "Segment length mismatch"
        else:
            print("Segmented dataset is empty.")

        # 3. Test AudioFileDataset (with Mel Spectrogram output)
        print("\n3. Testing AudioFileDataset (Mel Spectrogram output):")
        mel_params_test = {'n_fft': 1024, 'hop_length': 256, 'n_mels': 64}
        mel_dataset = AudioFileDataset(DUMMY_AUDIO_DIR, target_sr=SAMPLE_RATE,
                                   segment_length_samples=SEGMENT_SAMPLES,
                                   use_mel_spectrogram=True, mel_params=mel_params_test)
        if len(mel_dataset) > 0:
            sample_mel_spec, _ = mel_dataset[0]
            print(f"Sample Mel spectrogram shape: {sample_mel_spec.shape}") # (n_mels, time_frames)
            # Expected time_frames: SEGMENT_SAMPLES // hop_length + 1 (approx)
            expected_frames = SEGMENT_SAMPLES // mel_params_test['hop_length']
            # Librosa padding might add a frame or two
            assert sample_mel_spec.shape[0] == mel_params_test['n_mels'], "Mel band mismatch"
            assert abs(sample_mel_spec.shape[1] - expected_frames) <= 2 , f"Mel frames mismatch: got {sample_mel_spec.shape[1]}, expected around {expected_frames}"
        else:
            print("Mel dataset is empty.")


        # 4. Test DataLoader (with padding for variable length full files)
        print("\n4. Testing DataLoader (full files with padding):")
        # Note: For real training, segment_length_samples is usually preferred over padding full files
        # unless the model is designed for variable length input (e.g., with attention over all).
        full_file_loader = create_dataloader(DUMMY_AUDIO_DIR, batch_size=2, target_sr=SAMPLE_RATE,
                                             segment_length_samples=None, # Variable length
                                             num_workers=0)
        if len(full_file_loader) > 0:
            for batch_data, batch_indices in full_file_loader:
                print(f"Batch data shape (padded full files): {batch_data.shape}") # (batch_size, max_len_in_batch)
                print(f"Batch indices: {batch_indices}")
                break # Just check one batch
        else:
            print("Full file DataLoader is empty.")

        # 5. Test DataLoader (with fixed-size segments)
        print("\n5. Testing DataLoader (fixed-size segments):")
        segmented_loader = create_dataloader(DUMMY_AUDIO_DIR, batch_size=2, target_sr=SAMPLE_RATE,
                                             segment_length_samples=SEGMENT_SAMPLES, num_workers=0)
        if len(segmented_loader) > 0:
            for batch_data_seg, _ in segmented_loader:
                print(f"Batch data shape (segments): {batch_data_seg.shape}") # (batch_size, SEGMENT_SAMPLES)
                assert batch_data_seg.shape[1] == SEGMENT_SAMPLES, "Segment length mismatch in batch"
                break
        else:
            print("Segmented DataLoader is empty.")

        # 6. Test DataLoader (Mel Spectrograms, fixed size segments)
        print("\n6. Testing DataLoader (Mel Spectrograms, fixed size segments):")
        mel_loader = create_dataloader(DUMMY_AUDIO_DIR, batch_size=2, target_sr=SAMPLE_RATE,
                                       segment_length_samples=SEGMENT_SAMPLES,
                                       use_mel_spectrogram=True, mel_params=mel_params_test,
                                       num_workers=0)
        if len(mel_loader) > 0:
            for batch_mel_spec, _ in mel_loader:
                print(f"Batch Mel spectrogram shape: {batch_mel_spec.shape}")
                # (batch_size, n_mels, time_frames)
                assert batch_mel_spec.shape[1] == mel_params_test['n_mels'], "Mel band mismatch in batch"
                assert abs(batch_mel_spec.shape[2] - expected_frames) <=2 , "Mel frames mismatch in batch"
                break
        else:
            print("Mel Spectrogram DataLoader is empty.")


    except ImportError as e:
        print(f"Skipping some dataset tests due to missing torchaudio or other dependency: {e}")
    except Exception as e:
        print(f"An error occurred during dataset testing: {e}")
    finally:
        # Clean up dummy directory
        if os.path.exists(DUMMY_AUDIO_DIR):
            for i in range(3):
                dummy_file = os.path.join(DUMMY_AUDIO_DIR, f"dummy_{i}.wav")
                if os.path.exists(dummy_file):
                    os.remove(dummy_file)
            os.rmdir(DUMMY_AUDIO_DIR)
        print("\nDataset and DataLoader example finished.")
