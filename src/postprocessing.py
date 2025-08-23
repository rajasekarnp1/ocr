import numpy as np
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def apply_gain(audio_data: np.ndarray, gain_db: float) -> np.ndarray:
    """
    Applies a gain in decibels (dB) to the audio data.

    Args:
        audio_data (np.ndarray): Input audio signal.
        gain_db (float): Gain to apply in decibels.
                         Positive for amplification, negative for attenuation.

    Returns:
        np.ndarray: Audio signal with gain applied.
    """
    if not isinstance(audio_data, np.ndarray):
        logger.error("Input audio_data must be a NumPy array.")
        raise TypeError("Input audio_data must be a NumPy array.")

    gain_linear = 10**(gain_db / 20.0)
    processed_audio = audio_data * gain_linear
    logger.info(f"Applied gain of {gain_db:.2f} dB (linear: {gain_linear:.4f}). Max before: {np.max(np.abs(audio_data)):.4f}, Max after: {np.max(np.abs(processed_audio)):.4f}")
    return processed_audio

def clip_audio(audio_data: np.ndarray, min_val: float = -1.0, max_val: float = 1.0) -> tuple[np.ndarray, int]:
    """
    Clips the audio data to the specified minimum and maximum values.
    Also returns the number of samples that were clipped.

    Args:
        audio_data (np.ndarray): Input audio signal.
        min_val (float, optional): Minimum value to clip to. Defaults to -1.0.
        max_val (float, optional): Maximum value to clip to. Defaults to 1.0.

    Returns:
        tuple[np.ndarray, int]:
            - clipped_audio_data (np.ndarray): Audio signal clipped to [min_val, max_val].
            - num_clipped_samples (int): The number of samples that were clipped.
    """
    if not isinstance(audio_data, np.ndarray):
        logger.error("Input audio_data must be a NumPy array.")
        raise TypeError("Input audio_data must be a NumPy array.")

    original_max = np.max(audio_data) if audio_data.size > 0 else 0
    original_min = np.min(audio_data) if audio_data.size > 0 else 0

    clipped_audio = np.clip(audio_data, min_val, max_val)

    # Calculate number of clipped samples more accurately
    num_clipped_lower = np.sum(audio_data < min_val)
    num_clipped_upper = np.sum(audio_data > max_val)
    num_clipped_samples = int(num_clipped_lower + num_clipped_upper) # Ensure it's Python int

    if num_clipped_samples > 0:
        logger.warning(f"Clipping occurred: {num_clipped_samples} samples were clipped. ({num_clipped_lower} below {min_val}, {num_clipped_upper} above {max_val}). Max before clip: {original_max:.4f}, Min before clip: {original_min:.4f}")
    else:
        logger.info(f"No clipping was necessary. Original Min/Max: {original_min:.4f}/{original_max:.4f} within [{min_val}, {max_val}]")

    return clipped_audio, num_clipped_samples


if __name__ == '__main__':
    logger.info("Starting postprocessing.py example usage...")

    # Create a dummy audio signal
    sample_rate = 16000
    duration = 1.0
    frequency = 100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    original_signal_no_peaks = 0.7 * np.sin(2 * np.pi * frequency * t)

    # Add some samples that will exceed +/- 1.0 after gain
    original_signal_with_peaks = original_signal_no_peaks.copy()
    # These values are chosen so that after a 6dB gain (approx 2x), they will exceed 1.0 or -1.0
    original_signal_with_peaks[100] = 0.8
    original_signal_with_peaks[200] = -0.9

    logger.info(f"Original signal (with peaks) max: {np.max(original_signal_with_peaks):.4f}, min: {np.min(original_signal_with_peaks):.4f}")

    # Test apply_gain
    logger.info("\n--- Testing apply_gain ---")
    gain_to_apply_db = 6.0 # Amplify by approx 2x (linear gain = 1.995)
    gained_signal = apply_gain(original_signal_with_peaks.copy(), gain_to_apply_db)
    expected_linear_gain = 10**(gain_to_apply_db / 20.0)

    logger.info(f"Gained signal max: {np.max(gained_signal):.4f}, min: {np.min(gained_signal):.4f}")
    # Check a few points for correct gain application
    assert np.isclose(original_signal_with_peaks[0] * expected_linear_gain, gained_signal[0]), "Gain application test failed at point 0."
    assert np.isclose(original_signal_with_peaks[100] * expected_linear_gain, gained_signal[100]), "Gain application test failed at peak point 100." # Expected 1.596
    assert np.isclose(original_signal_with_peaks[200] * expected_linear_gain, gained_signal[200]), "Gain application test failed at peak point 200." # Expected -1.795


    # Test clip_audio
    logger.info("\n--- Testing clip_audio ---")
    # Use the gained signal which has peaks outside [-1, 1]
    # gained_signal[100] is approx 1.596
    # gained_signal[200] is approx -1.795
    clipped_signal, num_clipped = clip_audio(gained_signal.copy(), min_val=-1.0, max_val=1.0)
    logger.info(f"Clipped signal max: {np.max(clipped_signal):.4f}, min: {np.min(clipped_signal):.4f}")
    logger.info(f"Number of samples clipped: {num_clipped}")

    assert np.all(clipped_signal >= -1.0) and np.all(clipped_signal <= 1.0), "Clipping failed to limit values to [-1, 1]."
    # Check that the specific points were clipped
    assert clipped_signal[100] == 1.0, f"Expected sample 100 to be clipped to 1.0, got {clipped_signal[100]}"
    assert clipped_signal[200] == -1.0, f"Expected sample 200 to be clipped to -1.0, got {clipped_signal[200]}"
    # Only these two points were designed to be clipped
    # Other points might be clipped if original_signal_no_peaks * gain > 1.0
    # Max of original_signal_no_peaks is 0.7. 0.7 * 1.995 = 1.3965. So other points will be clipped too.
    # Let's count how many points in gained_signal were > 1 or < -1
    expected_num_clipped = np.sum(gained_signal > 1.0) + np.sum(gained_signal < -1.0)
    assert num_clipped == expected_num_clipped, f"Expected {expected_num_clipped} clipped samples based on gained signal, got {num_clipped}."


    # Test clipping with a signal already within bounds
    no_clip_signal, num_clipped_no_op = clip_audio(original_signal_no_peaks.copy()) # original_signal_no_peaks has max 0.7
    assert num_clipped_no_op == 0, "Clipping occurred for a signal already within bounds."
    assert np.array_equal(original_signal_no_peaks, no_clip_signal), "Signal within bounds was modified by clipping."
    logger.info("Clipping for signal within bounds handled correctly.")

    logger.info("postprocessing.py example usage finished.")
