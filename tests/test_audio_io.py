# tests/test_audio_io.py
import unittest
import os
import numpy as np
import soundfile as sf # Using soundfile directly for test audio creation

# Adjust import path based on your project structure.
# This assumes 'src' is in PYTHONPATH or tests are run from project root.
try:
    from src.audio_io import read_audio, write_audio
except ImportError:
    # Fallback for cases where src is not directly in PYTHONPATH
    # This might happen if tests are run from within the tests/ directory directly
    # without project-level test runners like pytest that adjust paths.
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.audio_io import read_audio, write_audio


class TestAudioIO(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory and dummy audio file for tests."""
        self.test_dir = "temp_test_audio_dir"
        os.makedirs(self.test_dir, exist_ok=True)

        self.dummy_audio_path = os.path.join(self.test_dir, "test_audio.wav")
        self.sample_rate = 44100
        self.duration = 0.1  # Short duration for fast tests
        self.frequency = 440

        # Create a simple sine wave audio file
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), False)
        self.original_data = 0.5 * np.sin(2 * np.pi * self.frequency * t)

        # Use soundfile to write the initial test audio
        try:
            sf.write(self.dummy_audio_path, self.original_data, self.sample_rate)
        except Exception as e:
            self.fail(f"Setup failed: Could not write dummy audio file using soundfile: {e}")

    def tearDown(self):
        """Clean up the temporary directory and files after tests."""
        if os.path.exists(self.dummy_audio_path):
            os.remove(self.dummy_audio_path)
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)

    def test_read_audio_basic(self):
        """Test basic audio reading functionality."""
        audio_data, sr = read_audio(self.dummy_audio_path)

        self.assertIsNotNone(audio_data, "Audio data should not be None.")
        self.assertIsInstance(audio_data, np.ndarray, "Audio data should be a NumPy array.")
        self.assertEqual(sr, self.sample_rate, f"Sampling rate should be {self.sample_rate}.")
        self.assertEqual(len(audio_data.shape), 1, "Audio data should be mono (1D array).")
        self.assertTrue(np.allclose(audio_data, self.original_data, atol=1e-4),
                        "Read audio data does not match original data closely enough.")

    def test_read_audio_resample(self):
        """Test audio reading with resampling."""
        target_sr = 16000
        audio_data, sr = read_audio(self.dummy_audio_path, target_sr=target_sr)

        self.assertIsNotNone(audio_data)
        self.assertEqual(sr, target_sr, f"Sampling rate should be resampled to {target_sr}.")

        # Expected length after resampling
        expected_len = int(self.duration * target_sr)
        self.assertAlmostEqual(len(audio_data), expected_len, delta=1, # librosa resampling might differ by 1 sample
                               msg="Length of resampled audio is incorrect.")

    def test_read_non_existent_file(self):
        """Test reading a non-existent audio file."""
        audio_data, sr = read_audio("non_existent_file.wav")
        self.assertIsNone(audio_data, "Audio data should be None for non-existent file.")
        self.assertIsNone(sr, "Sampling rate should be None for non-existent file.")

    def test_write_audio(self):
        """Test writing audio data to a file."""
        output_path = os.path.join(self.test_dir, "output_audio.wav")
        test_data = np.array([0.1, 0.2, 0.3, 0.2, 0.1], dtype=np.float32)
        test_sr = 22050

        write_audio(output_path, test_data, test_sr)
        self.assertTrue(os.path.exists(output_path), "Output audio file was not created.")

        # Verify the content by reading it back
        read_back_data, read_back_sr = sf.read(output_path, dtype='float32') # Use soundfile for direct check

        self.assertEqual(read_back_sr, test_sr, "Sampling rate of written file is incorrect.")
        self.assertTrue(np.allclose(read_back_data, test_data, atol=1e-6),
                        "Data in written file does not match original test data.")

        if os.path.exists(output_path):
            os.remove(output_path)

    def test_write_and_read_consistency(self):
        """Test that writing then reading an audio file preserves data."""
        path_to_rw_test = os.path.join(self.test_dir, "rw_test.wav")

        # Write using our function
        write_audio(path_to_rw_test, self.original_data, self.sample_rate)
        self.assertTrue(os.path.exists(path_to_rw_test))

        # Read back using our function
        loaded_data, loaded_sr = read_audio(path_to_rw_test)

        self.assertEqual(loaded_sr, self.sample_rate)
        self.assertIsNotNone(loaded_data)
        self.assertEqual(loaded_data.shape, self.original_data.shape)
        self.assertTrue(np.allclose(loaded_data, self.original_data, atol=1e-4), # Increased atol for potential minor changes due to write/read cycle
                        "Data read back does not match data written.")

        if os.path.exists(path_to_rw_test):
            os.remove(path_to_rw_test)

if __name__ == '__main__':
    # This allows running the tests directly from the command line
    # For more sophisticated test discovery and execution, use pytest or unittest test runners.
    # Example: `python -m unittest tests.test_audio_io` from the project root.
    unittest.main(verbosity=2)
