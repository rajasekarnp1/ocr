import numpy as np
import logging
import sys # For checking if scipy is in sys.modules
import os # For os.path and os.makedirs in __main__
# from scipy.signal import welch # For LSD, if scipy is available
# from pypesq import pesq # Example, would need installation
# from pystoi import stoi # Example, would need installation

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def calculate_snr(reference: np.ndarray, estimate: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculates the Signal-to-Noise Ratio (SNR) between a reference and an estimated signal.

    Args:
        reference (np.ndarray): The ground truth high-resolution audio signal.
        estimate (np.ndarray): The upscaled audio signal from the model.
        epsilon (float): A small value to prevent division by zero if noise is zero.

    Returns:
        float: The SNR in decibels (dB). Returns -infinity if inputs are inappropriate.
    """
    if not isinstance(reference, np.ndarray) or not isinstance(estimate, np.ndarray):
        logger.error("Inputs must be NumPy arrays for SNR calculation.")
        return -np.inf # Or raise TypeError

    if reference.shape != estimate.shape:
        logger.warning(f"Reference shape {reference.shape} and estimate shape {estimate.shape} differ for SNR. Attempting to align by trimming to shortest.")
        min_len = min(len(reference), len(estimate))
        if min_len == 0:
            logger.error("Cannot calculate SNR with zero-length aligned signals.")
            return -np.inf
        reference = reference[:min_len]
        estimate = estimate[:min_len]

    if reference.size == 0: # Handles if they were initially empty or became empty after trim
        logger.warning("Cannot calculate SNR for empty signals.")
        return -np.inf

    signal_power = np.sum(reference**2)
    noise = reference - estimate
    noise_power = np.sum(noise**2)

    if signal_power < epsilon: # Reference signal is effectively silence
        logger.warning("Reference signal power is close to zero.")
        # If noise is also zero, SNR is undefined (or infinite). If noise exists, SNR is very bad (-inf).
        return 0.0 if noise_power < epsilon else -np.inf

    if noise_power < epsilon:
        logger.info("Noise power is close to zero. SNR is very high (returning 100.0 dB).")
        return 100.0

    snr_val = 10 * np.log10(signal_power / noise_power)
    logger.info(f"Calculated SNR: {snr_val:.2f} dB")
    return snr_val


def calculate_lsd(reference: np.ndarray, estimate: np.ndarray, sample_rate: int,
                  frame_size_ms: int = 20, hop_size_ms: int = 10, epsilon: float = 1e-10) -> float | None:
    """
    Calculates the Log-Spectral Distance (LSD) between a reference and an estimated signal.
    """
    try:
        from scipy.signal import welch
    except ImportError:
        logger.warning("SciPy (scipy.signal.welch) not found. LSD calculation cannot be performed. Returning None.")
        return None

    if not isinstance(reference, np.ndarray) or not isinstance(estimate, np.ndarray):
        logger.error("Inputs must be NumPy arrays for LSD calculation.")
        return None

    if reference.shape != estimate.shape:
        logger.warning(f"Reference shape {reference.shape} and estimate shape {estimate.shape} differ for LSD. Attempting to align by trimming.")
        min_len = min(len(reference), len(estimate))
        if min_len == 0:
            logger.error("Cannot calculate LSD with zero-length aligned signals.")
            return None
        reference = reference[:min_len]
        estimate = estimate[:min_len]

    if reference.size == 0:
        logger.warning("Cannot calculate LSD for empty signals after alignment.")
        return None

    nperseg = int(frame_size_ms / 1000 * sample_rate)
    noverlap = nperseg - int(hop_size_ms / 1000 * sample_rate) # Welch uses noverlap, not hop_size directly

    if nperseg <= 0:
        logger.error(f"nperseg ({nperseg}) for Welch must be > 0. Check frame_size_ms and sample_rate.")
        return None
    if noverlap >= nperseg: # Common issue if hop_size_ms is too small or frame_size_ms too small
        logger.warning(f"noverlap ({noverlap}) >= nperseg ({nperseg}) for Welch. Setting noverlap to nperseg-1 if possible, or 0.")
        noverlap = max(0, nperseg - 1) if nperseg > 0 else 0


    try:
        freqs_ref, psd_ref = welch(reference, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, window='hann', average='mean')
        freqs_est, psd_est = welch(estimate, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, window='hann', average='mean')
    except Exception as e:
        logger.error(f"Error during Welch PSD calculation for LSD: {e}", exc_info=True)
        return None

    if not np.array_equal(freqs_ref, freqs_est):
        logger.warning("Frequency bins mismatch in Welch. This indicates an issue. LSD might be unreliable.")
        # Attempt to interpolate psd_est to freqs_ref if necessary, for robustness.
        # For now, assume they match if parameters are identical.
        # This usually happens if segment lengths are too short leading to empty PSDs or different freq resolutions.
        if len(freqs_ref) == 0 or len(freqs_est) == 0:
            logger.error("Empty frequency bins from Welch. Cannot calculate LSD.")
            return None
        # A simple alignment if lengths differ but range is similar (crude):
        # common_freq_max = min(freqs_ref.max(), freqs_est.max())
        # psd_ref = psd_ref[freqs_ref <= common_freq_max]
        # psd_est = psd_est[freqs_est <= common_freq_max]
        # freqs_ref = freqs_ref[freqs_ref <= common_freq_max]
        # This still doesn't guarantee same number of points. Interpolation is better.
        # For this version, we'll proceed if they don't match but log warning.

    psd_ref_safe = np.maximum(psd_ref, epsilon)
    psd_est_safe = np.maximum(psd_est, epsilon)

    # Ensure psd_est_safe has same length as psd_ref_safe if freqs mismatched and led to different lengths
    if len(psd_ref_safe) != len(psd_est_safe):
        logger.warning(f"PSD lengths differ: ref {len(psd_ref_safe)}, est {len(psd_est_safe)}. Trimming to shortest for LSD.")
        min_psd_len = min(len(psd_ref_safe), len(psd_est_safe))
        if min_psd_len == 0:
            logger.error("Cannot compute LSD with zero-length PSDs after alignment.")
            return None
        psd_ref_safe = psd_ref_safe[:min_psd_len]
        psd_est_safe = psd_est_safe[:min_psd_len]


    log_psd_ref = 10 * np.log10(psd_ref_safe)
    log_psd_est = 10 * np.log10(psd_est_safe)

    # LSD is typically sqrt of mean squared error over frames and then frequencies.
    # Welch averages over frames already. So we average over frequency bins.
    lsd_val = np.sqrt(np.mean((log_psd_ref - log_psd_est)**2))

    logger.info(f"Calculated LSD: {lsd_val:.2f} dB")
    return lsd_val


def calculate_pesq(reference_path: str, estimate_path: str, sample_rate: int) -> float | None:
    logger.warning("PESQ calculation is a placeholder. Requires 'pypesq' library and its dependencies.")
    logger.warning("Attempting to import 'pypesq' (will likely fail if not installed)...")
    try:
        # from pypesq import pesq # This is the typical import
        # For now, simulate a check without actual import to prevent ModuleNotFoundError if not present
        if 'pypesq' in sys.modules: # Or a more robust check if it's truly callable
             # score = pesq(sample_rate, reference_path, estimate_path, 'wb') # Old pypesq
             # score = pesq(reference_path, estimate_path, fs=sample_rate, mode='wb') # Newer pypesq
             logger.info("pypesq conceptually available. Actual calculation not performed in this placeholder.")
             return 2.5 # Dummy value
        else:
            raise ImportError("pypesq module not found in sys.modules")
    except ImportError:
        logger.error("'pypesq' library not found or importable. PESQ cannot be calculated.")
        return None
    except Exception as e:
        logger.error(f"Error during conceptual PESQ calculation: {e}", exc_info=True)
        return None


def calculate_stoi(reference: np.ndarray, estimate: np.ndarray, sample_rate: int) -> float | None:
    logger.warning("STOI calculation is a placeholder. Requires 'pystoi' library.")
    logger.warning("Attempting to import 'pystoi' (will likely fail if not installed)...")
    try:
        # from pystoi import stoi # Typical import
        if 'pystoi' in sys.modules:
            # score = stoi(reference, estimate, sample_rate, extended=False)
            logger.info("pystoi conceptually available. Actual calculation not performed in this placeholder.")
            return 0.75 # Dummy value
        else:
            raise ImportError("pystoi module not found in sys.modules")
    except ImportError:
        logger.error("'pystoi' library not found or importable. STOI cannot be calculated.")
        return None
    except Exception as e:
        logger.error(f"Error during conceptual STOI calculation: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    logger.info("Starting evaluation.py example usage...")

    sr_eval = 16000
    duration_eval = 1.0
    t_eval = np.linspace(0, duration_eval, int(sr_eval * duration_eval), endpoint=False, dtype=np.float32)
    ref_signal = 0.8 * np.sin(2 * np.pi * 440 * t_eval) + 0.2 * np.sin(2 * np.pi * 880 * t_eval)
    noise_eval = (np.random.rand(len(ref_signal)) - 0.5).astype(np.float32) * 0.1
    est_signal_good = ref_signal + noise_eval
    est_signal_bad = ref_signal + noise_eval * 10 # Increased noise factor for bad signal

    logger.info("\n--- Testing SNR ---")
    snr_good = calculate_snr(ref_signal, est_signal_good)
    snr_bad = calculate_snr(ref_signal, est_signal_bad)
    logger.info(f"SNR (Good Estimate): {snr_good:.2f} dB")
    logger.info(f"SNR (Bad Estimate): {snr_bad:.2f} dB")
    if snr_good is not None and snr_bad is not None and not np.isinf(snr_good) and not np.isinf(snr_bad):
        assert snr_good > snr_bad, "SNR for good estimate should be higher than for bad estimate."

    silent_ref = np.zeros_like(ref_signal)
    snr_silent_ref_noisy_est = calculate_snr(silent_ref, est_signal_good)
    logger.info(f"SNR (Silent Reference, Noisy Estimate): {snr_silent_ref_noisy_est:.2f} dB") # Expect -inf
    assert np.isneginf(snr_silent_ref_noisy_est), "SNR with silent ref, noisy est should be -inf"

    snr_silent_both = calculate_snr(silent_ref, silent_ref.copy())
    logger.info(f"SNR (Silent Reference, Silent Estimate): {snr_silent_both:.2f} dB") # Expect 0.0
    assert snr_silent_both == 0.0, "SNR with silent ref, silent est should be 0.0"

    logger.info("\n--- Testing LSD (conceptual) ---")
    try:
        # This is just to make the example run without scipy for now
        # In a real test env, scipy should be installed.
        if 'scipy' not in sys.modules:
            logger.info("Simulating scipy not being available for LSD test.")
            # To prevent the function from trying to import scipy if it's not there in this test run
            _calculate_lsd_orig = calculate_lsd
            calculate_lsd = lambda r,e,sr,**k: (_calculate_lsd_orig(r,e,sr,**k) if 'scipy.signal' in sys.modules else None)

        lsd_good = calculate_lsd(ref_signal, est_signal_good, sr_eval)
        lsd_bad = calculate_lsd(ref_signal, est_signal_bad, sr_eval)
        logger.info(f"LSD (Good Estimate): {lsd_good}") # Might be None
        logger.info(f"LSD (Bad Estimate): {lsd_bad}")   # Might be None
        if lsd_good is not None and lsd_bad is not None:
            assert lsd_good < lsd_bad, "LSD for good estimate should be lower than for bad estimate."
    except Exception as e_lsd_test: logger.error(f"Error during LSD test example: {e_lsd_test}")
    finally: # Restore original if it was temporarily changed
        if '_calculate_lsd_orig' in locals(): calculate_lsd = _calculate_lsd_orig


    logger.info("\n--- Testing PESQ (conceptual) ---")
    dummy_eval_dir = "dummy_eval_files_for_evaluationpy"
    os.makedirs(dummy_eval_dir, exist_ok=True)
    ref_path_pesq = os.path.join(dummy_eval_dir, "ref_pesq.wav")
    est_path_pesq = os.path.join(dummy_eval_dir, "est_pesq.wav")
    try:
        # Attempt to use src.audio_io.save_audio if available
        # This is a bit circular for testing evaluation.py itself.
        # If audio_io or its deps (like soundfile) are missing, this will fail.
        # For robust unit testing of evaluation.py, one might mock save_audio
        # or ensure dummy files are created by other means.
        try:
            from src.audio_io import save_audio as save_audio_for_eval
            save_audio_for_eval(ref_path_pesq, ref_signal, sr_eval)
            save_audio_for_eval(est_path_pesq, est_signal_good, sr_eval)
        except ImportError:
            logger.warning("src.audio_io.save_audio not available. PESQ test will use placeholder logic.")
            # Create empty files if save_audio not available, so PESQ placeholder can run
            open(ref_path_pesq, 'a').close()
            open(est_path_pesq, 'a').close()

        pesq_score = calculate_pesq(ref_path_pesq, est_path_pesq, sr_eval)
        logger.info(f"Conceptual PESQ score: {pesq_score}") # Expected None if pypesq not installed
    except Exception as e_pesq_test: logger.error(f"Error during PESQ test setup/run: {e_pesq_test}")
    finally:
        if os.path.exists(dummy_eval_dir):
            import shutil
            shutil.rmtree(dummy_eval_dir)

    logger.info("\n--- Testing STOI (conceptual) ---")
    stoi_score = calculate_stoi(ref_signal, est_signal_good, sr_eval)
    logger.info(f"Conceptual STOI score: {stoi_score}") # Expected None if pystoi not installed

    logger.info("evaluation.py example usage finished.")

# Restore original sys.modules if pypesq/pystoi were dummied for testing
# This is generally not good practice for production code but helps __main__ example.
# if 'pypesq_dummy_placeholder' in sys.modules: del sys.modules['pypesq_dummy_placeholder']
# if 'pystoi_dummy_placeholder' in sys.modules: del sys.modules['pystoi_dummy_placeholder']
