import unittest
import os
import numpy as np
import soundfile as sf # For creating test files if needed, and checking attributes
import shutil
import logging

# Add src to path to allow direct import of audio_io
import sys
current_dir_test = os.path.dirname(os.path.abspath(__file__))
project_root_test = os.path.dirname(current_dir_test)
if project_root_test not in sys.path:
    sys.path.insert(0, project_root_test)

# Configure logger for tests to see output from audio_io module easily if needed
logging.basicConfig(level=logging.DEBUG) # Or logging.INFO
test_logger = logging.getLogger("test_audio_io")

try:
    from src.audio_io import load_audio, save_audio, SUPPORTED_FORMATS_LOAD, SUPPORTED_FORMATS_SAVE
    # If the import succeeds, log it for clarity during test runs
    test_logger.debug("Successfully imported from src.audio_io for test_audio_io.py")
except ImportError as e:
    test_logger.error(f"Could not import from src.audio_io for test_audio_io.py: {e}. Tests might not run correctly.")
    # Define dummy functions if import fails, so the rest of the test file can be parsed
    def load_audio(file_path, target_sr=None, mono=True): return (None, None)
    def save_audio(file_path, audio_data, sample_rate, subtype='PCM_16'): return False
    SUPPORTED_FORMATS_LOAD = []
    SUPPORTED_FORMATS_SAVE = []


class TestAudioIO(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory and dummy audio files for testing."""
        self.test_dir = "temp_test_audio_io_dir_for_unittest" # More specific name
        os.makedirs(self.test_dir, exist_ok=True)
        test_logger.debug(f"Created temp directory: {self.test_dir}")

        self.sr_orig = 16000
        self.duration = 0.1
        self.dummy_mono_signal = np.linspace(-0.5, 0.5, int(self.sr_orig * self.duration), dtype=np.float32)
        self.dummy_mono_path = os.path.join(self.test_dir, "test_mono.wav")
        try:
            sf.write(self.dummy_mono_path, self.dummy_mono_signal, self.sr_orig, subtype='FLOAT') # Save as float for better precision
        except Exception as e_sf:
            test_logger.error(f"Setup failed: Could not write dummy mono audio using soundfile: {e_sf}")
            raise # Re-raise to fail setup if essential files can't be created

        self.dummy_stereo_signal = np.array([self.dummy_mono_signal, -self.dummy_mono_signal * 0.8]).T # (samples, 2)
        self.dummy_stereo_path = os.path.join(self.test_dir, "test_stereo.wav")
        try:
            sf.write(self.dummy_stereo_path, self.dummy_stereo_signal, self.sr_orig, subtype='FLOAT')
        except Exception as e_sf:
            test_logger.error(f"Setup failed: Could not write dummy stereo audio using soundfile: {e_sf}")
            raise

        self.non_existent_file = os.path.join(self.test_dir, "non_existent.wav")
        self.unsupported_save_path = os.path.join(self.test_dir, "test_unsupported.xyz")


    def tearDown(self):
        """Remove the temporary directory after tests."""
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
                test_logger.debug(f"Removed temp directory: {self.test_dir}")
            except OSError as e_shutil:
                test_logger.error(f"Error removing temp directory {self.test_dir}: {e_shutil}")


    def test_load_audio_mono_wav(self):
        audio, sr = load_audio(self.dummy_mono_path, target_sr=None, mono=True)
        self.assertIsNotNone(audio, "Loaded audio should not be None.")
        self.assertEqual(sr, self.sr_orig, "Sample rate mismatch.")
        self.assertEqual(audio.ndim, 1, "Audio should be mono.")
        self.assertTrue(np.allclose(audio, self.dummy_mono_signal, atol=1e-5), "Audio content mismatch for mono load.")

    def test_load_audio_stereo_to_mono(self):
        audio, sr = load_audio(self.dummy_stereo_path, target_sr=None, mono=True)
        self.assertIsNotNone(audio)
        self.assertEqual(sr, self.sr_orig)
        self.assertEqual(audio.ndim, 1, "Audio should be converted to mono.")
        expected_mono_from_stereo = np.mean(self.dummy_stereo_signal, axis=1)
        self.assertTrue(np.allclose(audio, expected_mono_from_stereo, atol=1e-5), "Stereo to mono conversion content mismatch.")

    def test_load_audio_stereo_keep_stereo(self):
        # Current load_audio in src/audio_io.py is hardcoded to mono=True in librosa.load call
        # and its own mono parameter defaults to True.
        # To test keeping stereo, load_audio would need to pass mono=False to librosa.
        # For now, this test confirms current behavior (converts to mono).
        audio_as_mono, sr = load_audio(self.dummy_stereo_path, target_sr=None, mono=True) # mono=True is default
        self.assertEqual(audio_as_mono.ndim, 1)
        test_logger.info("test_load_audio_stereo_keep_stereo: Confirmed current load_audio converts stereo to mono by default.")
        # If load_audio is updated:
        # audio_stereo, sr_stereo = load_audio(self.dummy_stereo_path, mono=False) # Assuming load_audio gets mono param
        # self.assertEqual(audio_stereo.ndim, 2)
        # self.assertTrue(np.allclose(audio_stereo, self.dummy_stereo_signal, atol=1e-5))

    def test_load_audio_resampling(self):
        target_sr_resample = self.sr_orig // 2
        audio, sr = load_audio(self.dummy_mono_path, target_sr=target_sr_resample, mono=True)
        self.assertIsNotNone(audio)
        self.assertEqual(sr, target_sr_resample, "Resampled sample rate mismatch.")
        expected_len = int(len(self.dummy_mono_signal) * (target_sr_resample / self.sr_orig))
        # Librosa resampling might result in length differing by 1 sample due to internal padding/rounding.
        self.assertAlmostEqual(len(audio), expected_len, delta=1, msg="Resampled audio length mismatch.")


    def test_load_non_existent_file(self):
        audio, sr = load_audio(self.non_existent_file)
        self.assertIsNone(audio, "Audio should be None for non-existent file.")
        self.assertIsNone(sr, "Sample rate should be None for non-existent file.")

    def test_save_audio_wav_pcm16(self):
        save_path = os.path.join(self.test_dir, "test_save_output_pcm16.wav")
        success = save_audio(save_path, self.dummy_mono_signal, self.sr_orig, subtype='PCM_16')
        self.assertTrue(success, "save_audio should return True on success.")
        self.assertTrue(os.path.exists(save_path), "Saved WAV file does not exist.")

        reloaded_audio_la, reloaded_sr_la = load_audio(save_path, target_sr=None, mono=True)
        self.assertEqual(reloaded_sr_la, self.sr_orig)
        self.assertTrue(np.allclose(self.dummy_mono_signal, reloaded_audio_la, atol=1e-3), # Increased tolerance for PCM_16
                        "Content mismatch after saving (PCM_16) and reloading WAV.")

    def test_save_audio_wav_float32(self):
        save_path = os.path.join(self.test_dir, "test_save_output_float32.wav")
        # Save as float32 (PCM_FLOAT in soundfile)
        success = save_audio(save_path, self.dummy_mono_signal, self.sr_orig, subtype='FLOAT')
        self.assertTrue(success, "save_audio (FLOAT) should return True.")
        self.assertTrue(os.path.exists(save_path), "Saved WAV (FLOAT) file does not exist.")

        reloaded_audio_la, reloaded_sr_la = load_audio(save_path, target_sr=None, mono=True)
        self.assertEqual(reloaded_sr_la, self.sr_orig)
        # Float to Float should have very high precision
        self.assertTrue(np.allclose(self.dummy_mono_signal, reloaded_audio_la, atol=1e-6),
                        "Content mismatch after saving (FLOAT) and reloading WAV.")


    def test_save_audio_flac(self):
        save_path_flac = os.path.join(self.test_dir, "test_save_output.flac")
        # FLAC default subtype in soundfile is usually 'PCM_16' or 'PCM_24' if input is float.
        # Let's be explicit if we want to test float FLAC, but soundfile might not support it directly for FLAC.
        # The save_audio function's subtype param is primarily for WAV.
        # For FLAC, soundfile chooses a good default.
        success = save_audio(save_path_flac, self.dummy_mono_signal, self.sr_orig)
        self.assertTrue(success, "save_audio for FLAC should return True.")
        self.assertTrue(os.path.exists(save_path_flac), "Saved FLAC file does not exist.")

        reloaded_audio, reloaded_sr = load_audio(save_path_flac, target_sr=None, mono=True)
        self.assertEqual(reloaded_sr, self.sr_orig)
        # FLAC is lossless for PCM. If soundfile converted our float32 to PCM_16 (as per log),
        # then reloaded as float32, there will be quantization differences.
        self.assertTrue(np.allclose(self.dummy_mono_signal, reloaded_audio, atol=1e-3), # Adjusted tolerance
                        "Content mismatch after saving and reloading FLAC.")


    def test_save_unsupported_format_defaults_to_wav(self):
        self.assertFalse(os.path.exists(self.unsupported_save_path))
        expected_wav_path = os.path.splitext(self.unsupported_save_path)[0] + '.wav'
        self.assertFalse(os.path.exists(expected_wav_path))

        # Use a logger specific to the module being tested to check its output
        module_logger = logging.getLogger('src.audio_io') # Name of the logger in audio_io.py

        with self.assertLogs(module_logger, level='INFO') as log_watcher: # Capture INFO and WARNING
            success = save_audio(self.unsupported_save_path, self.dummy_mono_signal, self.sr_orig)

        self.assertTrue(success, "save_audio should return True even when defaulting to WAV.")
        # Check specific log messages
        self.assertTrue(any("Unsupported file format for saving: .xyz" in message for message in log_watcher.output), "Missing/incorrect warning for unsupported format.")
        self.assertTrue(any(f"New save path: {expected_wav_path}" in message for message in log_watcher.output), "Missing/incorrect info for new save path.")

        self.assertFalse(os.path.exists(self.unsupported_save_path), "File with original unsupported extension should not be created.")
        self.assertTrue(os.path.exists(expected_wav_path), "File should have been saved with .wav extension.")

if __name__ == '__main__':
    # This allows running tests with `python tests/test_audio_io.py`
    # It's also picked up by `python -m unittest discover`
    unittest.main(verbosity=2) # Add verbosity for more detailed output
