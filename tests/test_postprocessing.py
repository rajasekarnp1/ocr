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
logging.basicConfig(level=logging.DEBUG)
test_logger = logging.getLogger("test_postprocessing")

try:
    from src.postprocessing import apply_gain, clip_audio
    test_logger.debug("Successfully imported from src.postprocessing for test_postprocessing.py")
except ImportError:
    test_logger.error("Warning: Could not import from src.postprocessing for test_postprocessing.py.")
    # Define dummy functions if import fails
    def apply_gain(audio_data, gain_db): return audio_data
    def clip_audio(audio_data, min_val=-1.0, max_val=1.0): return audio_data, 0


class TestPostprocessing(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 16000
        self.duration = 0.1 # Short duration for tests
        self.frequency = 440
        self.t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False, dtype=np.float32)
        self.base_signal = 0.5 * np.sin(2 * np.pi * self.frequency * self.t) # Peak is 0.5
        test_logger.debug("TestPostprocessing setUp completed.")

    def test_apply_gain_positive_db(self):
        """Test applying positive gain (amplification)."""
        gain_db = 6.0  # Approx 2x linear gain (1.99526)
        gained_signal = apply_gain(self.base_signal.copy(), gain_db)
        expected_peak = 0.5 * (10**(6.0/20.0))
        self.assertAlmostEqual(np.max(np.abs(gained_signal)), expected_peak, places=5,
                               msg="Positive gain application resulted in unexpected peak.")

    def test_apply_gain_negative_db(self):
        """Test applying negative gain (attenuation)."""
        gain_db = -6.0 # Approx 0.5x linear gain (0.501187)
        attenuated_signal = apply_gain(self.base_signal.copy(), gain_db)
        expected_peak = 0.5 * (10**(-6.0/20.0))
        self.assertAlmostEqual(np.max(np.abs(attenuated_signal)), expected_peak, places=5,
                               msg="Negative gain application resulted in unexpected peak.")

    def test_apply_gain_zero_db(self):
        """Test applying zero gain (no change)."""
        no_change_signal = apply_gain(self.base_signal.copy(), 0.0)
        self.assertTrue(np.allclose(self.base_signal, no_change_signal, atol=1e-7),
                        "Applying 0dB gain modified the signal.")

    def test_clip_audio_no_clipping(self):
        """Test clipping when signal is already within bounds."""
        # base_signal has peak 0.5, so it's within [-1, 1]
        clipped_signal, num_clipped = clip_audio(self.base_signal.copy(), min_val=-1.0, max_val=1.0)
        self.assertTrue(np.array_equal(self.base_signal, clipped_signal),
                        "Signal within bounds was modified by clipping.")
        self.assertEqual(num_clipped, 0, "num_clipped should be 0 when no clipping occurs.")

    def test_clip_audio_positive_clipping(self):
        """Test clipping for values exceeding max_val."""
        signal_to_clip = self.base_signal.copy() * 3.0 # Peak becomes 1.5
        clipped_signal, num_clipped = clip_audio(signal_to_clip, min_val=-1.0, max_val=1.0)

        self.assertLessEqual(np.max(clipped_signal), 1.0 + 1e-7, "Max value exceeds 1.0 after clipping.") # Add epsilon for float comparisons
        self.assertGreater(num_clipped, 0, "num_clipped should be > 0 when positive clipping occurs.")
        # Verify that values that were > 1.0 are now 1.0
        self.assertTrue(np.all(np.isclose(clipped_signal[signal_to_clip > 1.0], 1.0)),
                        "Values > 1.0 were not clipped to 1.0")


    def test_clip_audio_negative_clipping(self):
        """Test clipping for values below min_val."""
        signal_to_clip = self.base_signal.copy() * 3.0 # Min becomes -1.5
        clipped_signal, num_clipped = clip_audio(signal_to_clip, min_val=-1.0, max_val=1.0)
        self.assertGreaterEqual(np.min(clipped_signal), -1.0 - 1e-7, "Min value is less than -1.0 after clipping.") # Add epsilon
        self.assertGreater(num_clipped, 0, "num_clipped should be > 0 when negative clipping occurs.")
        self.assertTrue(np.all(np.isclose(clipped_signal[signal_to_clip < -1.0], -1.0)),
                        "Values < -1.0 were not clipped to -1.0")

    def test_clip_audio_both_sides_clipping(self):
        """Test clipping for values exceeding both max_val and min_val."""
        signal_to_clip = self.base_signal.copy() # Peak is 0.5
        signal_to_clip[0] = 2.0  # Exceeds max_val
        signal_to_clip[1] = -2.0 # Exceeds min_val

        module_logger = logging.getLogger('src.postprocessing')
        with self.assertLogs(module_logger, level='WARNING') as log_watcher:
            clipped_signal, num_clipped = clip_audio(signal_to_clip, min_val=-0.8, max_val=0.8)

        self.assertLessEqual(np.max(clipped_signal), 0.8 + 1e-7)
        self.assertGreaterEqual(np.min(clipped_signal), -0.8 - 1e-7)
        # Only two samples were deliberately set outside custom bounds for this specific test.
        # Other samples of base_signal (max 0.5) are within [-0.8, 0.8].
        self.assertEqual(num_clipped, 2, f"Expected 2 samples to be clipped, got {num_clipped}.")
        self.assertTrue(any("Clipping occurred" in message for message in log_watcher.output))


    def test_clip_audio_custom_bounds(self):
        """Test clipping with custom min_val and max_val."""
        custom_min, custom_max = -0.3, 0.3
        signal_to_clip = self.base_signal.copy() # Original peak 0.5

        clipped_signal, num_clipped = clip_audio(signal_to_clip, min_val=custom_min, max_val=custom_max)
        self.assertLessEqual(np.max(clipped_signal), custom_max + 1e-7, "Max value after custom clip exceeds target.")
        self.assertGreaterEqual(np.min(clipped_signal), custom_min - 1e-7, "Min value after custom clip is below target.")
        self.assertGreater(num_clipped, 0, "Expected clipping with custom bounds.")

    def test_clip_audio_empty_input(self):
        """Test clip_audio with an empty numpy array."""
        empty_signal = np.array([], dtype=np.float32)
        clipped_signal, num_clipped = clip_audio(empty_signal)
        self.assertEqual(len(clipped_signal), 0, "Clipping empty signal should result in an empty signal.")
        self.assertEqual(num_clipped, 0, "Number of clipped samples should be 0 for empty input.")


if __name__ == '__main__':
    # If running the test file directly, configure logging for this run.
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    unittest.main(verbosity=2)
