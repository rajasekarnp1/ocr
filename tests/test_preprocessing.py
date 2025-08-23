import unittest
import numpy as np
import logging

# Add src to path to allow direct import
import sys
import os
current_dir_test = os.path.dirname(os.path.abspath(__file__))
project_root_test = os.path.dirname(current_dir_test)
if project_root_test not in sys.path:
    sys.path.insert(0, project_root_test)

# Configure initial basic logging for the test file itself.
# The module being tested (src.preprocessing) should have its own logger.
logging.basicConfig(level=logging.DEBUG) # So we can see logs from the module if needed.
test_logger = logging.getLogger("test_preprocessing")


try:
    from src.preprocessing import peak_normalize, rms_normalize, frame_audio
    test_logger.debug("Successfully imported from src.preprocessing for test_preprocessing.py")
except ImportError as e:
    test_logger.error(f"Could not import from src.preprocessing for test_preprocessing.py: {e}")
    # Define dummy functions if import fails
    def peak_normalize(audio_data, target_peak=0.95): return audio_data
    def rms_normalize(audio_data, target_rms_dbfs=-20.0): return audio_data
    def frame_audio(audio_data, frame_size, hop_size, pad_end=True):
        if not isinstance(audio_data, np.ndarray) or audio_data.ndim != 1:
            raise ValueError("audio_data must be a 1D NumPy array.")
        if audio_data.size == 0: return np.array([]).reshape(0, frame_size)
        # Simplified dummy for parsing, real tests need real function.
        num_frames = (len(audio_data) - frame_size) // hop_size + 1 if len(audio_data) >= frame_size else 1
        if num_frames <=0 : num_frames = 1 # Ensure at least one frame for dummy
        return np.zeros((num_frames if num_frames > 0 else 1, frame_size))


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 16000
        self.duration = 0.5
        self.frequency = 100
        self.t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False, dtype=np.float32)
        self.signal = 0.5 * np.sin(2 * np.pi * self.frequency * self.t)

        self.signal_high_peak = self.signal.copy()
        self.signal_high_peak[0] = 0.8
        self.signal_high_peak[1] = -0.8

        self.silent_signal = np.zeros(100, dtype=np.float32)
        test_logger.debug("TestPreprocessing setUp completed.")


    def test_peak_normalize_typical_case(self):
        target_peak = 0.9
        normalized = peak_normalize(self.signal_high_peak.copy(), target_peak=target_peak)
        self.assertAlmostEqual(np.max(np.abs(normalized)), target_peak, places=5,
                               msg="Peak normalization did not reach target peak.")

        signal_already_high = self.signal_high_peak.copy() * 2.0
        normalized_down = peak_normalize(signal_already_high, target_peak=0.5)
        self.assertAlmostEqual(np.max(np.abs(normalized_down)), 0.5, places=5,
                               msg="Peak normalization did not scale down correctly.")

    def test_peak_normalize_silent_input(self):
        normalized = peak_normalize(self.silent_signal.copy(), target_peak=0.9)
        self.assertTrue(np.all(normalized == 0), "Peak normalization of silent signal should be silent.")

    def test_peak_normalize_target_one(self):
        normalized = peak_normalize(self.signal.copy(), target_peak=1.0)
        self.assertAlmostEqual(np.max(np.abs(normalized)), 1.0, places=5)

    def test_peak_normalize_invalid_target(self):
        with self.assertRaisesRegex(ValueError, "target_peak must be positive"):
            peak_normalize(self.signal.copy(), target_peak=0)
        with self.assertRaisesRegex(ValueError, "target_peak must be positive"):
            peak_normalize(self.signal.copy(), target_peak=-0.5)

    def test_rms_normalize_typical_case(self):
        test_signal = self.signal.copy()
        # Ensure signal is not silent for RMS test; if self.signal can be silent, adjust.
        if np.sqrt(np.mean(test_signal**2)) < 1e-9: # if it's silent
             test_signal = test_signal + 0.001 # make it not silent (e.g. add DC offset)
             if np.sqrt(np.mean(test_signal**2)) < 1e-9: # if still silent (e.g. was all zeros)
                  test_signal[0] = 0.1 # Force a non-zero value

        target_dbfs = -18.0
        normalized = rms_normalize(test_signal, target_rms_dbfs=target_dbfs)

        rms_linear_normalized = np.sqrt(np.mean(normalized**2))
        # Add epsilon to prevent log10(0) if signal became silent (should not with good input)
        rms_dbfs_normalized = 20 * np.log10(rms_linear_normalized + 1e-10)

        self.assertAlmostEqual(rms_dbfs_normalized, target_dbfs, delta=0.1, # RMS matching can have small deviations
                               msg=f"RMS norm failed. Target: {target_dbfs}dBFS, Got: {rms_dbfs_normalized:.2f}dBFS")

    def test_rms_normalize_silent_input(self):
        normalized = rms_normalize(self.silent_signal.copy(), target_rms_dbfs=-20.0)
        self.assertTrue(np.all(normalized == 0), "RMS normalization of silent signal should be silent.")

    def test_rms_normalize_empty_input(self):
        empty_arr = np.array([], dtype=np.float32)
        normalized = rms_normalize(empty_arr.copy())
        self.assertEqual(len(normalized), 0, "RMS normalization of empty signal should be empty.")


    def test_frame_audio_basic(self):
        signal = np.arange(20, dtype=np.float32)
        frames = frame_audio(signal, frame_size=5, hop_size=2, pad_end=False)
        self.assertEqual(frames.shape, (8, 5))
        self.assertTrue(np.array_equal(frames[0], np.array([0, 1, 2, 3, 4])))
        self.assertTrue(np.array_equal(frames[-1], np.array([14, 15, 16, 17, 18])))

    def test_frame_audio_with_padding(self):
        signal = np.arange(20, dtype=np.float32)
        frame_size = 5
        hop_size = 3
        # Expected based on src/preprocessing.py (corrected logic from previous run):
        # (N-1)*hop + frame_size = (6-1)*3 + 5 = 15+5=20. No padding needed. Num frames = 6.
        frames = frame_audio(signal, frame_size=frame_size, hop_size=hop_size, pad_end=True)
        self.assertEqual(frames.shape, (6, frame_size))
        self.assertTrue(np.array_equal(frames[-1], np.array([15, 16, 17, 18, 19])))

        signal_short_for_hop = np.arange(19, dtype=np.float32)
        # Expected based on src/preprocessing.py:
        # num_total_frames = (19-5)//3 + 1 = 4+1=5. Check if last frame reaches end: (5-1)*3+5 = 17 < 19. So num_total_frames = 5+1 = 6.
        # Padded length = (6-1)*3+5 = 20. Padding = 1.
        frames_padded = frame_audio(signal_short_for_hop, frame_size=frame_size, hop_size=hop_size, pad_end=True)
        self.assertEqual(frames_padded.shape, (6, frame_size))
        self.assertTrue(np.array_equal(frames_padded[-1], np.array([15, 16, 17, 18, 0])))


    def test_frame_audio_signal_shorter_than_frame(self):
        signal = np.array([1, 2, 3], dtype=np.float32)
        frames_padded = frame_audio(signal, frame_size=5, hop_size=2, pad_end=True)
        self.assertEqual(frames_padded.shape, (1, 5))
        self.assertTrue(np.array_equal(frames_padded[0], np.array([1, 2, 3, 0, 0])))

        frames_no_pad_short = frame_audio(signal, frame_size=5, hop_size=2, pad_end=False)
        self.assertEqual(frames_no_pad_short.shape, (1,5))
        self.assertTrue(np.array_equal(frames_no_pad_short[0], np.array([1,2,3,0,0])))


    def test_frame_audio_empty_signal(self):
        signal = np.array([], dtype=np.float32)
        frames = frame_audio(signal, frame_size=5, hop_size=2, pad_end=True)
        self.assertEqual(frames.shape, (0, 5))
        frames_no_pad = frame_audio(signal, frame_size=5, hop_size=2, pad_end=False)
        self.assertEqual(frames_no_pad.shape, (0, 5))

    def test_frame_audio_hop_greater_than_frame(self):
        signal = np.arange(20, dtype=np.float32)
        module_logger = logging.getLogger('src.preprocessing')
        with self.assertLogs(module_logger, level='WARNING') as log_watcher:
            frames = frame_audio(signal, frame_size=3, hop_size=5, pad_end=False)
        self.assertTrue(any("hop_size is greater than frame_size" in message for message in log_watcher.output))
        self.assertEqual(frames.shape, (4,3))
        self.assertTrue(np.array_equal(frames[0], np.array([0,1,2])))
        self.assertTrue(np.array_equal(frames[1], np.array([5,6,7])))


if __name__ == '__main__':
    # Configure logging for tests if run directly
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Using unittest's own runner is preferred.
    unittest.main(verbosity=2) # Verbosity for more detailed output from tests.
