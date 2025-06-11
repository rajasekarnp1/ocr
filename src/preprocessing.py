import numpy as np
import logging

# Configure basic logging for the module
logger = logging.getLogger(__name__)
if not logger.handlers: # Avoid adding multiple handlers if reloaded
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def peak_normalize(audio_data: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Normalizes the audio data to a target peak amplitude.

    Args:
        audio_data (np.ndarray): The input audio signal.
        target_peak (float, optional): The desired peak amplitude. Defaults to 0.95.
                                       Should be positive.

    Returns:
        np.ndarray: The peak-normalized audio signal.
    """
    if not isinstance(audio_data, np.ndarray):
        logger.error("Input audio_data must be a NumPy array.")
        raise TypeError("Input audio_data must be a NumPy array.")
    if target_peak <= 0:
        logger.error("target_peak must be positive.")
        raise ValueError("target_peak must be positive.")

    current_peak = np.max(np.abs(audio_data))

    if current_peak == 0: # Audio is silent
        logger.info("Audio is silent, no peak normalization applied.")
        return audio_data.copy() # Return a copy to maintain consistency with non-silent path

    gain = target_peak / current_peak
    normalized_audio = audio_data * gain

    # Clip to prevent exceeding target_peak due to floating point inaccuracies, though unlikely with this method.
    # normalized_audio = np.clip(normalized_audio, -target_peak, target_peak)

    logger.info(f"Peak normalization applied. Original peak: {current_peak:.4f}, Target peak: {target_peak:.4f}, Gain: {gain:.4f}")
    return normalized_audio

def rms_normalize(audio_data: np.ndarray, target_rms_dbfs: float = -20.0) -> np.ndarray:
    """
    Normalizes the audio data to a target RMS level in dBFS.

    Args:
        audio_data (np.ndarray): The input audio signal.
        target_rms_dbfs (float, optional): The desired RMS level in dBFS.
                                         Common values range from -12 to -26 dBFS.
                                         Defaults to -20.0 dBFS.

    Returns:
        np.ndarray: The RMS-normalized audio signal.
    """
    if not isinstance(audio_data, np.ndarray):
        logger.error("Input audio_data must be a NumPy array.")
        raise TypeError("Input audio_data must be a NumPy array.")

    if audio_data.size == 0:
        logger.warning("Input audio_data is empty. Returning empty array.")
        return audio_data.copy() # Return a copy

    # Calculate current RMS in linear scale
    current_rms_linear = np.sqrt(np.mean(audio_data**2))

    # Using a very small number to check for silence (epsilon comparison)
    if current_rms_linear < 1e-9: # Essentially silent or very close to zero
        logger.info("Audio is silent or near silent, no RMS normalization applied to avoid extreme gain.")
        return audio_data.copy() # Return a copy

    # Convert target RMS from dBFS to linear scale
    target_rms_linear = 10**(target_rms_dbfs / 20.0)

    gain = target_rms_linear / current_rms_linear
    normalized_audio = audio_data * gain

    # Optional: Check new RMS for debugging
    # new_rms_linear_check = np.sqrt(np.mean(normalized_audio**2))
    # logger.debug(f"Target RMS: {target_rms_dbfs:.2f} dBFS ({target_rms_linear:.4f} linear). Original RMS: {20*np.log10(current_rms_linear):.2f} dBFS. New RMS: {20*np.log10(new_rms_linear_check):.2f} dBFS. Gain: {gain:.4f}")

    logger.info(f"RMS normalization applied. Target RMS: {target_rms_dbfs:.2f} dBFS. Gain: {gain:.4f}")
    return normalized_audio

def frame_audio(audio_data: np.ndarray, frame_size: int, hop_size: int, pad_end: bool = True) -> np.ndarray:
    """
    Splits audio data into frames.

    Args:
        audio_data (np.ndarray): The input audio signal (1D).
        frame_size (int): The number of samples per frame.
        hop_size (int): The number of samples to advance between frames.
        pad_end (bool, optional): Whether to pad the end of the audio signal
                                  to ensure all samples are included in at least one frame.
                                  If True, pads with zeros. Defaults to True.

    Returns:
        np.ndarray: A 2D array where each row is a frame. Shape: (num_frames, frame_size).
    """
    if not isinstance(audio_data, np.ndarray) or audio_data.ndim != 1:
        logger.error("audio_data must be a 1D NumPy array.")
        raise ValueError("audio_data must be a 1D NumPy array.")
    if frame_size <= 0 or hop_size <= 0:
        logger.error("frame_size and hop_size must be positive.")
        raise ValueError("frame_size and hop_size must be positive.")
    if hop_size > frame_size:
        logger.warning("hop_size is greater than frame_size. This will result in gaps between frames.")

    num_samples = len(audio_data)

    if pad_end:
        if num_samples > 0: # Only pad if there's data
            # Calculate padding needed so the last frame starts at or before the last sample
            # and includes the last sample.
            # Number of frames needed if we don't pad: (num_samples - frame_size) // hop_size + 1 (only if num_samples >= frame_size)
            # If num_samples < frame_size, we need 1 frame.
            if num_samples <= frame_size:
                num_total_frames = 1
            else:
                # Calculate total frames needed to cover all samples with current hop_size
                num_total_frames = (num_samples - frame_size) // hop_size + 1
                # If the last frame doesn't reach the end of the signal, add another frame
                if (num_total_frames - 1) * hop_size + frame_size < num_samples:
                    num_total_frames +=1

            if num_samples > 0 : # Only proceed if there are samples
                 if num_total_frames > 0: # Should always be true if num_samples > 0
                     required_len = (num_total_frames - 1) * hop_size + frame_size
                     padding = max(0, required_len - num_samples)
                 else: # num_samples == 0
                     padding = 0

                 if padding > 0:
                    audio_data_padded = np.pad(audio_data, (0, padding), 'constant', constant_values=0)
                    num_samples = len(audio_data_padded) # Update num_samples after padding
                    logger.debug(f"Padded audio with {padding} zeros. New length: {num_samples}")
                    effective_audio_data = audio_data_padded
                 else:
                    effective_audio_data = audio_data.copy() # Use a copy if no padding applied but pad_end is true
            else:
                effective_audio_data = audio_data.copy()
        else: # num_samples == 0
            effective_audio_data = audio_data.copy() # Empty array
    else: # No padding at the end
        effective_audio_data = audio_data.copy()

    # Update num_samples based on effective_audio_data which might be padded or a copy
    num_samples = len(effective_audio_data)


    if num_samples == 0:
        logger.info("Audio is empty, returning zero frames.")
        return np.array([]).reshape(0, frame_size)

    if num_samples < frame_size:
        logger.warning(f"Audio length ({num_samples}) is less than frame_size ({frame_size}). A single, padded frame will be returned.")
        # Pad the single frame to frame_size
        padded_frame = np.pad(effective_audio_data, (0, frame_size - num_samples), 'constant', constant_values=0)
        return padded_frame.reshape(1, frame_size)

    # Calculate the number of frames using as_strided
    # This calculation is for when num_samples >= frame_size
    num_frames = (num_samples - frame_size) // hop_size + 1

    # Using as_strided creates a view, not a copy, which is memory efficient.
    shape = (num_frames, frame_size)
    strides = (effective_audio_data.strides[0] * hop_size, effective_audio_data.strides[0])
    frames = np.lib.stride_tricks.as_strided(effective_audio_data, shape=shape, strides=strides)

    logger.info(f"Framed audio into {num_frames} frames. Frame size: {frame_size}, Hop size: {hop_size}.")
    return frames


if __name__ == '__main__':
    logger.info("Starting preprocessing.py example usage...")

    # Create a dummy audio signal (e.g., sine wave)
    sample_rate = 16000
    duration = 1.0
    frequency = 100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Signal with DC offset and varying amplitude
    original_signal_float = 0.2 * np.sin(2 * np.pi * frequency * t) + 0.1 * np.sin(2 * np.pi * frequency * 2 * t) + 0.05
    original_signal_int16 = (original_signal_float * 32767).astype(np.int16)

    # --- Test Peak Normalization ---
    logger.info("\n--- Testing Peak Normalization ---")
    target_peak_val = 0.8

    # Float signal
    normalized_peak_float = peak_normalize(original_signal_float.copy(), target_peak=target_peak_val)
    actual_peak_float = np.max(np.abs(normalized_peak_float))
    logger.info(f"Float signal: Original peak: {np.max(np.abs(original_signal_float)):.4f}, Normalized peak: {actual_peak_float:.4f} (Target: {target_peak_val})")
    assert np.isclose(actual_peak_float, target_peak_val), "Peak normalization failed for float signal."

    # Int signal (results will be float after normalization)
    # Convert int to float in range [-1, 1] for processing
    processed_int_signal = original_signal_int16.astype(np.float32) / np.iinfo(np.int16).max
    normalized_peak_int_as_float = peak_normalize(processed_int_signal, target_peak=target_peak_val)
    actual_peak_int_as_float = np.max(np.abs(normalized_peak_int_as_float))
    logger.info(f"Int signal (processed as float): Original peak (scaled): {np.max(np.abs(processed_int_signal)):.4f}, Normalized peak: {actual_peak_int_as_float:.4f} (Target: {target_peak_val})")
    assert np.isclose(actual_peak_int_as_float, target_peak_val), "Peak normalization failed for int signal (processed as float)."

    # Silent signal
    silent_signal = np.zeros(100)
    normalized_silent = peak_normalize(silent_signal.copy(), target_peak=target_peak_val)
    assert np.all(normalized_silent == 0), "Peak normalization of silent signal should result in a silent signal."
    logger.info("Peak normalization of silent signal handled correctly.")

    # --- Test RMS Normalization ---
    logger.info("\n--- Testing RMS Normalization ---")
    target_rms_val_dbfs = -23.0
    target_rms_val_linear = 10**(target_rms_val_dbfs / 20.0)

    # Float signal
    normalized_rms_float = rms_normalize(original_signal_float.copy(), target_rms_dbfs=target_rms_val_dbfs)
    actual_rms_float_linear = np.sqrt(np.mean(normalized_rms_float**2))
    logger.info(f"Float signal: Original RMS: {20*np.log10(np.sqrt(np.mean(original_signal_float**2)) + 1e-10):.2f} dBFS, Normalized RMS: {20*np.log10(actual_rms_float_linear + 1e-10):.2f} dBFS (Target: {target_rms_val_dbfs:.2f} dBFS)")
    assert np.isclose(actual_rms_float_linear, target_rms_val_linear, atol=1e-3), f"RMS normalization failed for float signal. Expected linear: {target_rms_val_linear}, Got: {actual_rms_float_linear}"

    # Int signal (results will be float)
    normalized_rms_int_as_float = rms_normalize(processed_int_signal, target_rms_dbfs=target_rms_val_dbfs)
    actual_rms_int_as_float_linear = np.sqrt(np.mean(normalized_rms_int_as_float**2))
    logger.info(f"Int signal (processed as float): Original RMS (scaled): {20*np.log10(np.sqrt(np.mean(processed_int_signal**2)) + 1e-10):.2f} dBFS, Normalized RMS: {20*np.log10(actual_rms_int_as_float_linear + 1e-10):.2f} dBFS (Target: {target_rms_val_dbfs:.2f} dBFS)")
    assert np.isclose(actual_rms_int_as_float_linear, target_rms_val_linear, atol=1e-3), f"RMS normalization failed for int signal. Expected linear: {target_rms_val_linear}, Got: {actual_rms_int_as_float_linear}"

    # Silent signal
    normalized_silent_rms = rms_normalize(silent_signal.copy(), target_rms_dbfs=target_rms_val_dbfs)
    assert np.all(normalized_silent_rms == 0), "RMS normalization of silent signal should result in a silent signal."
    logger.info("RMS normalization of silent signal handled correctly.")

    # --- Test Framing ---
    logger.info("\n--- Testing Audio Framing ---")
    test_signal = np.arange(20, dtype=np.float32) # Simple signal for easy verification
    frame_sz = 5
    hop_sz = 2

    # Test 1: Basic framing, no end padding (last samples might be dropped)
    frames1 = frame_audio(test_signal, frame_size=frame_sz, hop_size=hop_sz, pad_end=False)
    logger.info(f"Frames1 (pad_end=False, len={len(test_signal)}):\n{frames1}")
    # Expected: (20 - 5) // 2 + 1 = 15 // 2 + 1 = 7 + 1 = 8 frames
    # Last frame starts at (8-1)*2 = 14. Data: test_signal[14:19] = [14, 15, 16, 17, 18]
    assert frames1.shape == (8, frame_sz)
    assert np.array_equal(frames1[0], np.array([0, 1, 2, 3, 4]))
    assert np.array_equal(frames1[-1], np.array([14, 15, 16, 17, 18]))

    # Test 2: Framing with padding at the end
    frames2 = frame_audio(test_signal, frame_size=frame_sz, hop_size=hop_sz, pad_end=True)
    logger.info(f"Frames2 (pad_end=True, len={len(test_signal)}):\n{frames2}")
    # Original length 20. Last sample at index 19.
    # Last frame should contain sample 19.
    # num_total_frames = (20 - 1) // 2 + 1 = 19 // 2 + 1 = 9 + 1 = 10 frames
    # Padded length: (10-1)*2 + 5 = 18 + 5 = 23. Padding = 3.
    # Padded signal: [0,...,19,0] (since padding=1 for length 21)
    # Last frame starts at (9-1)*2 = 16. Data: test_signal_padded[16:21] = [16,17,18,19,0]
    assert frames2.shape == (9, frame_sz)
    assert np.array_equal(frames2[-1], np.array([16, 17, 18, 19, 0]))

    # Test 3: Signal shorter than frame_size, with padding
    short_signal = np.array([1, 2, 3], dtype=np.float32)
    frames3 = frame_audio(short_signal, frame_size=frame_sz, hop_size=hop_sz, pad_end=True)
    logger.info(f"Frames3 (short_signal, pad_end=True, len={len(short_signal)}):\n{frames3}")
    assert frames3.shape == (1, frame_sz)
    assert np.array_equal(frames3[0], np.array([1, 2, 3, 0, 0]))

    # Test 4: Signal shorter than frame_size, no padding
    # Current implementation will pad the frame itself to frame_size.
    frames4 = frame_audio(short_signal, frame_size=frame_sz, hop_size=hop_sz, pad_end=False)
    logger.info(f"Frames4 (short_signal, pad_end=False, len={len(short_signal)}):\n{frames4}")
    assert frames4.shape == (1, frame_sz)
    assert np.array_equal(frames4[0], np.array([1,2,3,0,0]))

    # Test 5: Empty signal
    empty_signal = np.array([], dtype=np.float32)
    frames5 = frame_audio(empty_signal, frame_size=frame_sz, hop_size=hop_sz, pad_end=True)
    logger.info(f"Frames5 (empty_signal, len={len(empty_signal)}):\n{frames5}")
    assert frames5.shape == (0, frame_sz)

    logger.info("preprocessing.py example usage finished.")
