import librosa
import soundfile as sf
import numpy as np
import logging
import os

# Configure basic logging for the module
logger = logging.getLogger(__name__)
if not logger.handlers: # Avoid adding multiple handlers if reloaded
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

SUPPORTED_FORMATS_LOAD = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']
SUPPORTED_FORMATS_SAVE = ['.wav', '.flac'] # Soundfile supports others, but WAV/FLAC are common for quality

def load_audio(file_path: str, target_sr: int = None, mono: bool = True) -> tuple[np.ndarray | None, int | None]:
    """
    Loads an audio file from the given path.

    Args:
        file_path (str): The path to the audio file.
        target_sr (int, optional): The target sample rate to resample the audio to.
                                   If None, uses the original sample rate. Defaults to None.
        mono (bool, optional): Whether to convert the audio to mono. Defaults to True.

    Returns:
        tuple[np.ndarray | None, int | None]: A tuple containing:
            - audio_data (np.ndarray): The loaded audio data as a NumPy array. None if loading fails.
            - sample_rate (int): The sample rate of the loaded audio. None if loading fails.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None, None

    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext not in SUPPORTED_FORMATS_LOAD:
        logger.warning(f"Unsupported file format for loading: {file_ext}. Attempting to load anyway.")
        # Librosa might still handle it, so we don't strictly prevent it here.

    try:
        # y is the audio time series, sr is the sample rate
        y, sr = librosa.load(file_path, sr=target_sr, mono=mono)
        logger.info(f"Successfully loaded audio from: {file_path} (Original SR: {librosa.get_samplerate(file_path)}, Target SR: {sr}, Mono: {mono})")
        return y, sr
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}", exc_info=True)
        return None, None

def save_audio(file_path: str, audio_data: np.ndarray, sample_rate: int, subtype: str = 'PCM_16') -> bool:
    """
    Saves audio data to a file.

    Args:
        file_path (str): The path to save the audio file.
        audio_data (np.ndarray): The audio data to save.
        sample_rate (int): The sample rate of the audio data.
        subtype (str, optional): The subtype for WAV file saving (e.g., 'PCM_16', 'PCM_24', 'FLOAT').
                                 Defaults to 'PCM_16'.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext not in SUPPORTED_FORMATS_SAVE:
        logger.warning(f"Unsupported file format for saving: {file_ext}. Defaulting to .wav")
        file_path = os.path.splitext(file_path)[0] + '.wav'
        logger.info(f"New save path: {file_path}")

    try:
        # Ensure directory exists
        # Check if the path is just a filename or includes a directory
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            logger.info(f"Created directory: {dir_name}")

        sf.write(file_path, audio_data, sample_rate, subtype=subtype)
        logger.info(f"Successfully saved audio to: {file_path} (SR: {sample_rate}, Subtype: {subtype})")
        return True
    except Exception as e:
        logger.error(f"Error saving audio file {file_path}: {e}", exc_info=True)
        return False

if __name__ == '__main__':
    # Create a dummy directory for testing if it doesn't exist
    dummy_dir = "dummy_audio_test_dir"
    if not os.path.exists(dummy_dir):
        os.makedirs(dummy_dir)
        logger.info(f"Created directory for testing: {dummy_dir}")

    logger.info("Starting audio_io.py example usage...")

    # Example: Create and save a dummy WAV file
    sample_rate_orig = 44100
    duration = 1.0  # seconds
    frequency = 440  # Hz (A4 note)
    t = np.linspace(0, duration, int(sample_rate_orig * duration), endpoint=False)
    dummy_signal = 0.5 * np.sin(2 * np.pi * frequency * t)

    dummy_file_path = os.path.join(dummy_dir, "dummy_audio.wav")

    logger.info(f"Attempting to save dummy signal to {dummy_file_path}...")
    save_successful = save_audio(dummy_file_path, dummy_signal, sample_rate_orig)

    if save_successful:
        logger.info("Dummy audio saved successfully.")

        # Example: Load the dummy WAV file
        logger.info(f"Attempting to load dummy signal from {dummy_file_path}...")
        loaded_signal, loaded_sr = load_audio(dummy_file_path)
        if loaded_signal is not None:
            logger.info(f"Loaded audio signal shape: {loaded_signal.shape}, Sample rate: {loaded_sr}")
            assert loaded_sr == sample_rate_orig, "Sample rate mismatch after loading."
            # Basic check for signal similarity
            if np.allclose(dummy_signal, loaded_signal[:len(dummy_signal)], atol=1e-4):
                 logger.info("Original and loaded signals are close enough.")
            else:
                 logger.warning("Original and loaded signals differ more than expected.")


        # Example: Load with resampling
        target_sr_resample = 16000
        logger.info(f"Attempting to load dummy signal with resampling to {target_sr_resample} Hz...")
        resampled_signal, resampled_sr = load_audio(dummy_file_path, target_sr=target_sr_resample)
        if resampled_signal is not None:
            logger.info(f"Resampled audio signal shape: {resampled_signal.shape}, Sample rate: {resampled_sr}")
            assert resampled_sr == target_sr_resample, "Resampled sample rate does not match target."

        # Example: Load non-existent file
        logger.info("Attempting to load non_existent_audio.wav (expected to fail)...")
        non_existent_signal, non_existent_sr = load_audio("non_existent_audio.wav")
        assert non_existent_signal is None and non_existent_sr is None, "Loading non-existent file did not return None."
        logger.info("Loading non-existent file handled as expected.")

        # Clean up the dummy file
        try:
            os.remove(dummy_file_path)
            logger.info(f"Cleaned up dummy file: {dummy_file_path}")
        except OSError as e:
            logger.error(f"Error removing dummy file {dummy_file_path}: {e}")

    else:
        logger.error("Failed to save the dummy audio file, skipping load tests.")

    # Clean up dummy directory if empty
    try:
        if os.path.exists(dummy_dir) and not os.listdir(dummy_dir): # Check if directory is empty
            os.rmdir(dummy_dir)
            logger.info(f"Cleaned up dummy directory: {dummy_dir}")
        elif os.path.exists(dummy_dir):
            logger.info(f"Dummy directory {dummy_dir} not empty, not removing.")
    except OSError as e:
        logger.error(f"Error removing dummy directory {dummy_dir}: {e}")

    logger.info("audio_io.py example usage finished.")
