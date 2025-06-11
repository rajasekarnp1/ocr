import torch
from torch.utils.data import Dataset, DataLoader
import os
import random
import logging
import numpy as np # Added for np.pad and np.random.randn if needed for dummy data

# Assuming audio_io and preprocessing are in src, and project is structured to allow this import.
# This might require running with python -m src.training.dataset or proper PYTHONPATH.
try:
    from ..audio_io import load_audio
except ImportError:
    logger = logging.getLogger(__name__) # Ensure logger is defined for this block
    logger.error("Failed to import load_audio from ..audio_io. Using dummy load_audio for dataset.py.")
    # Dummy load_audio if the import fails (e.g. torch/librosa not installed)
    def load_audio(file_path: str, target_sr: int = None, mono: bool = True) -> tuple[np.ndarray | None, int | None]:
        logger.warning(f"Using DUMMY load_audio for {file_path}. Returns random noise.")
        if target_sr is None: target_sr = 16000 # Default dummy SR
        # Return random noise of approx 1 second length as a NumPy array
        return np.random.randn(target_sr).astype(np.float32), target_sr


logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class PairedAudioDataset(Dataset):
    """
    Dataset for loading pairs of low-resolution and high-resolution audio files.
    Assumes that files with the same name in lr_dir and hr_dir form a pair.
    """
    def __init__(self,
                 lr_dir: str,
                 hr_dir: str,
                 target_sr: int,
                 segment_length: int = None,
                 normalize: bool = False, # Changed default to False as no normalize_fn is passed yet
                 ):
        super().__init__()
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.target_sr = target_sr
        self.segment_length = segment_length
        self.normalize = normalize

        try:
            lr_files_all = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir) if os.path.isfile(os.path.join(lr_dir, f))])
            hr_files_all = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if os.path.isfile(os.path.join(hr_dir, f))])
        except FileNotFoundError as e:
            logger.error(f"Directory not found: {e.filename}. Dataset will be empty.")
            self.file_pairs = []
            return


        lr_basenames = {os.path.basename(f):f for f in lr_files_all}
        hr_basenames = {os.path.basename(f):f for f in hr_files_all}

        self.file_pairs = []
        common_basenames_sorted = sorted(list(set(lr_basenames.keys()).intersection(set(hr_basenames.keys()))))

        if not common_basenames_sorted:
            logger.warning(f"No matching audio file pairs found in {lr_dir} and {hr_dir}. Dataset will be empty.")
            return

        for basename in common_basenames_sorted:
            self.file_pairs.append({
                'lr': lr_basenames[basename],
                'hr': hr_basenames[basename]
            })

        logger.info(f"Initialized PairedAudioDataset with {len(self.file_pairs)} pairs from {lr_dir} and {hr_dir}.")
        if len(self.file_pairs) < len(lr_files_all) or len(self.file_pairs) < len(hr_files_all):
            logger.warning("Some files did not form pairs and were excluded.")


    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        if idx >= len(self.file_pairs): # Should not happen with standard samplers
            raise IndexError("Index out of bounds")

        pair = self.file_pairs[idx]
        lr_path = pair['lr']
        hr_path = pair['hr']

        hr_audio, hr_sr = load_audio(hr_path, target_sr=self.target_sr, mono=True)
        lr_audio, lr_sr = load_audio(lr_path, target_sr=self.target_sr, mono=True)

        if hr_audio is None or lr_audio is None:
            logger.warning(f"Failed to load audio for pair: LR='{lr_path}', HR='{hr_path}'. Trying a different random item.")
            # Avoid infinite recursion if all files are bad or dataset is very small.
            # If this happens frequently, __init__ should pre-filter valid files.
            if len(self) > 1: # Only try random if there are other items
                return self.__getitem__(random.choice(list(set(range(len(self))) - {idx})))
            else: # Cannot recover if this is the only item or dataset is empty
                # This should ideally be caught by an empty dataset check in create_dataloader
                # Or return a specific error / dummy data that collate_fn can filter
                logger.error(f"Cannot recover from load failure for {lr_path}, {hr_path} with dataset size {len(self)}.")
                # Returning zero tensors as a fallback. Collate fn might need to handle this.
                # Or raise an error to stop training if data is critical.
                dummy_data = torch.zeros(1, self.segment_length if self.segment_length else self.target_sr)
                return {'low_res': dummy_data, 'high_res': dummy_data, 'lr_path': lr_path, 'hr_path': hr_path, 'error': True}


        if self.segment_length:
            min_len = min(len(hr_audio), len(lr_audio))
            if min_len == 0: # Handle case where audio loaded but is empty
                 logger.warning(f"Empty audio array loaded for {lr_path} or {hr_path}. Using zeros.")
                 hr_segment = np.zeros(self.segment_length, dtype=np.float32)
                 lr_segment = np.zeros(self.segment_length, dtype=np.float32)
            elif min_len < self.segment_length:
                pad_hr = self.segment_length - len(hr_audio) if len(hr_audio) < self.segment_length else 0
                pad_lr = self.segment_length - len(lr_audio) if len(lr_audio) < self.segment_length else 0

                if pad_hr > 0: hr_audio = np.pad(hr_audio, (0, pad_hr), 'constant')
                if pad_lr > 0: lr_audio = np.pad(lr_audio, (0, pad_lr), 'constant')
                start_idx = 0
                hr_segment = hr_audio
                lr_segment = lr_audio
            else:
                start_idx = random.randint(0, min_len - self.segment_length)
                hr_segment = hr_audio[start_idx : start_idx + self.segment_length]
                lr_segment = lr_audio[start_idx : start_idx + self.segment_length]
        else:
            hr_segment = hr_audio
            lr_segment = lr_audio
            if self.segment_length is None and hr_segment.shape[-1] != lr_segment.shape[-1]:
                 logger.warning(f"Full files loaded but lengths differ for {lr_path} ({lr_segment.shape[-1]}) and {hr_path} ({hr_segment.shape[-1]}). Collate will need to handle this.")


        # Placeholder for actual normalization if self.normalize is True
        # if self.normalize:
        #    from ..preprocessing import peak_normalize # Example
        #    hr_segment = peak_normalize(hr_segment)
        #    lr_segment = peak_normalize(lr_segment)

        hr_tensor = torch.from_numpy(hr_segment.copy()).float().unsqueeze(0)
        lr_tensor = torch.from_numpy(lr_segment.copy()).float().unsqueeze(0)

        return {'low_res': lr_tensor, 'high_res': hr_tensor, 'lr_path': lr_path, 'hr_path': hr_path}


def create_dataloader(lr_dir: str, hr_dir: str, target_sr: int, segment_length: int,
                      batch_size: int, num_workers: int = 0, shuffle: bool = True,
                      pin_memory: bool = False, drop_last: bool = True, **kwargs) -> DataLoader:
    dataset = PairedAudioDataset(lr_dir=lr_dir, hr_dir=hr_dir, target_sr=target_sr,
                                 segment_length=segment_length, **kwargs)

    if len(dataset) == 0:
        logger.error("Dataset is empty. Cannot create DataLoader.")
        return None

    # Basic collate_fn: default collate should work if all tensors are same shape.
    # If segment_length is None (full files), a custom collate_fn for padding is needed.
    # For now, assume segment_length is always provided for consistent tensor shapes.
    collate_function = None
    if segment_length is None:
        logger.warning("segment_length is None in create_dataloader. Default collate may fail if audio lengths vary.")
        # Define or import a custom collate_fn that pads sequences to max length in batch if needed.
        # def custom_collate(batch): ...
        # collate_function = custom_collate

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last,
                      collate_fn=collate_function)


if __name__ == '__main__':
    logger.info("Starting dataset.py example usage...")

    dummy_root = "dummy_dataset_test_fordatasetpy" # Unique name
    dummy_lr_dir = os.path.join(dummy_root, "lr")
    dummy_hr_dir = os.path.join(dummy_root, "hr")
    os.makedirs(dummy_lr_dir, exist_ok=True)
    os.makedirs(dummy_hr_dir, exist_ok=True)

    sr = 16000
    seg_len_samples = sr * 1

    # Create empty placeholder files for testing dataset logic (pairing, etc.)
    # Actual audio loading will use the DUMMY load_audio if torch/librosa is missing.
    file_basenames = ["sample1.wav", "sample2.wav", "sample3.flac"]
    for basename in file_basenames:
        open(os.path.join(dummy_lr_dir, basename), 'a').close()
        # Create HR pairs for some, not all, to test pairing logic
        if basename != "sample3.flac": # sample3.flac will be LR only
            open(os.path.join(dummy_hr_dir, basename), 'a').close()
    open(os.path.join(dummy_hr_dir, "only_hr.wav"), 'a').close() # HR only file

    logger.info(f"Created dummy files in {dummy_lr_dir} and {dummy_hr_dir}")

    try:
        paired_dataset = PairedAudioDataset(lr_dir=dummy_lr_dir, hr_dir=dummy_hr_dir,
                                            target_sr=sr, segment_length=seg_len_samples)
        logger.info(f"Number of file pairs found: {len(paired_dataset)}")
        # We expect "sample1.wav" and "sample2.wav" to form pairs.
        assert len(paired_dataset) == 2, f"Expected 2 file pairs, found {len(paired_dataset)}"

        if len(paired_dataset) > 0:
            logger.info("Attempting to get a sample item from the dataset...")
            # This will use the DUMMY load_audio if imports failed.
            sample_item = paired_dataset[0]
            logger.info(f"Sample item: LR path='{sample_item['lr_path']}', HR path='{sample_item['hr_path']}'")
            logger.info(f"  LR shape: {sample_item['low_res'].shape}, HR shape: {sample_item['high_res'].shape}")
            assert sample_item['low_res'].shape == (1, seg_len_samples), "Low-res segment shape incorrect."
            assert sample_item['high_res'].shape == (1, seg_len_samples), "High-res segment shape incorrect."

        logger.info("\n--- Testing DataLoader ---")
        dataloader = create_dataloader(lr_dir=dummy_lr_dir, hr_dir=dummy_hr_dir, target_sr=sr,
                                       segment_length=seg_len_samples, batch_size=1)
        if dataloader:
            logger.info("DataLoader created.")
            try:
                for i, batch in enumerate(dataloader):
                    logger.info(f"Batch {i+1}: LR shape={batch['low_res'].shape}, HR shape={batch['high_res'].shape}")
                    assert batch['low_res'].ndim == 3 and batch['low_res'].shape[0] == 1
                    assert batch['high_res'].ndim == 3 and batch['high_res'].shape[0] == 1
                    if i >= 0: break # Check only first batch
                logger.info("Iterating through DataLoader successful (first batch).")
            except Exception as e_dl:
                 logger.error(f"Error iterating DataLoader: {e_dl}. This might be due to dummy load_audio or other issues if torch is missing.")
        else:
            logger.error("DataLoader creation failed.")

    except ImportError as e_import: # Catch potential torch/librosa import errors
        logger.error(f"ImportError in dataset example: {e_import}. This means torch or related libraries are not installed.")
    except Exception as e_main:
        logger.error(f"An error occurred in dataset example: {e_main}", exc_info=True)
    finally:
        import shutil
        if os.path.exists(dummy_root):
            try:
                shutil.rmtree(dummy_root)
                logger.info(f"Cleaned up dummy dataset directory: {dummy_root}")
            except OSError as e_clean:
                logger.error(f"Error removing dummy dataset directory {dummy_root}: {e_clean}")

    logger.info("dataset.py example usage finished.")
