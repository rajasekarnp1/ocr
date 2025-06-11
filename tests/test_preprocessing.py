# tests/test_preprocessing.py
import unittest
import numpy as np
import librosa # For reference calculations if needed, or creating test data

# Adjust import path
try:
    from src.preprocessing import normalize_audio, get_mel_spectrogram, frame_audio
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.preprocessing import normalize_audio, get_mel_spectrogram, frame_audio

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        """Set up common test data."""
        self.sample_rate = 44100
        self.duration = 1.0  # 1 second
        self.frequency = 100 # A low frequency for easier spectrogram visualization if debugging

        # Basic sine wave
        self.test_audio_mono = 0.5 * np.sin(2 * np.pi * self.frequency *
                                            np.linspace(0, self.duration, int(self.sample_rate * self.duration), False))

        # Stereo audio (though most functions here might expect mono)
        self.test_audio_stereo = np.array([self.test_audio_mono, self.test_audio_mono * 0.8]).T # (samples, channels)

        self.silent_audio = np.zeros(self.sample_rate)
        self.empty_audio = np.array([])


    def test_normalize_audio_basic(self):
        """Test basic audio normalization."""
        target_db = -10.0
        normalized_audio = normalize_audio(self.test_audio_mono.copy(), target_db=target_db)

        self.assertIsNotNone(normalized_audio)
        self.assertEqual(normalized_audio.shape, self.test_audio_mono.shape)

        # Calculate RMS of normalized audio
        rms_normalized = np.sqrt(np.mean(normalized_audio**2))
        expected_rms = 10**(target_db / 20.0)

        self.assertAlmostEqual(rms_normalized, expected_rms, delta=1e-3, # Allow small delta
                               msg=f"Normalized RMS ({rms_normalized:.4f}) not close to target RMS ({expected_rms:.4f}).")
        self.assertTrue(np.max(np.abs(normalized_audio)) <= 1.0, "Normalized audio should be clipped to [-1, 1].")

    def test_normalize_audio_silent(self):
        """Test normalization of silent audio."""
        normalized_silent = normalize_audio(self.silent_audio.copy(), target_db=-20.0)
        self.assertTrue(np.all(normalized_silent == 0), "Normalizing silent audio should result in silent audio.")

    def test_normalize_audio_empty(self):
        """Test normalization of empty audio."""
        normalized_empty = normalize_audio(self.empty_audio.copy(), target_db=-20.0)
        self.assertEqual(len(normalized_empty), 0, "Normalizing empty audio should result in empty audio.")

    def test_normalize_already_normalized_audio(self):
        """Test normalizing audio that's already at target RMS (or very close)."""
        target_db = -20.0
        # First, normalize it
        once_normalized = normalize_audio(self.test_audio_mono.copy(), target_db=target_db)
        # Then, normalize it again to the same target
        twice_normalized = normalize_audio(once_normalized.copy(), target_db=target_db)

        rms_twice_normalized = np.sqrt(np.mean(twice_normalized**2))
        expected_rms = 10**(target_db / 20.0)
        self.assertAlmostEqual(rms_twice_normalized, expected_rms, delta=1e-4) # Should be very close now
        self.assertTrue(np.allclose(once_normalized, twice_normalized, atol=1e-5),
                        "Normalizing an already normalized audio changed it significantly.")


    def test_get_mel_spectrogram_basic(self):
        """Test basic mel spectrogram generation."""
        n_fft = 2048
        hop_length = 512
        n_mels = 128

        mel_spec = get_mel_spectrogram(self.test_audio_mono, sr=self.sample_rate,
                                       n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

        self.assertIsNotNone(mel_spec)
        self.assertIsInstance(mel_spec, np.ndarray)
        self.assertEqual(mel_spec.shape[0], n_mels, f"Number of mel bands should be {n_mels}.")

        # Expected number of time frames
        expected_frames = int(np.floor(len(self.test_audio_mono) / hop_length)) + 1
        # librosa's padding behavior might add frames, so allow some flexibility
        self.assertAlmostEqual(mel_spec.shape[1], expected_frames, delta=2, # Increased delta for librosa padding
                               msg=f"Number of time frames mismatch. Got {mel_spec.shape[1]}, expected around {expected_frames}.")

    def test_get_mel_spectrogram_params(self):
        """Test mel spectrogram with different parameters."""
        mel_spec_64 = get_mel_spectrogram(self.test_audio_mono, sr=self.sample_rate, n_mels=64, fmax=8000)
        self.assertEqual(mel_spec_64.shape[0], 64)

        # Test with a very small audio clip (shorter than n_fft)
        short_audio = self.test_audio_mono[:1000] # Approx 22ms, n_fft default is 2048 samples (46ms)
        mel_spec_short = get_mel_spectrogram(short_audio, sr=self.sample_rate, n_fft=2048, hop_length=512, n_mels=128)
        self.assertIsNotNone(mel_spec_short)
        self.assertEqual(mel_spec_short.shape[0], 128)
        # Even for short audio, librosa pads and computes at least one frame if center=True (default for melspectrogram)
        self.assertTrue(mel_spec_short.shape[1] > 0, "Mel spectrogram of short audio should have at least one frame.")


    def test_frame_audio_basic(self):
        """Test basic audio framing."""
        frame_size = 1024
        hop_size = 512

        frames = frame_audio(self.test_audio_mono, frame_size=frame_size, hop_size=hop_size)

        self.assertIsNotNone(frames)
        self.assertIsInstance(frames, np.ndarray)

        expected_num_frames = 1 + int(np.floor((len(self.test_audio_mono) - frame_size) / hop_size))
        self.assertEqual(frames.shape[0], expected_num_frames, "Number of frames is incorrect.")
        self.assertEqual(frames.shape[1], frame_size, "Frame size is incorrect.")

        # Check content of a frame (e.g., the first frame)
        self.assertTrue(np.array_equal(frames[0], self.test_audio_mono[:frame_size]),
                        "First frame content does not match original audio segment.")
        # Check content of a subsequent frame, considering the hop
        if expected_num_frames > 1:
             self.assertTrue(np.array_equal(frames[1], self.test_audio_mono[hop_size : hop_size + frame_size]),
                             "Second frame content does not match original audio segment with hop.")

    def test_frame_audio_short_input(self):
        """Test framing with audio shorter than frame_size."""
        frame_size = 2048
        hop_size = 512 # Does not matter much if audio is shorter than frame_size
        short_audio = self.test_audio_mono[:frame_size // 2]

        frames = frame_audio(short_audio, frame_size=frame_size, hop_size=hop_size)

        # The current implementation pads to make at least one frame
        self.assertEqual(frames.shape[0], 1, "Should produce one frame for short audio (padded).")
        self.assertEqual(frames.shape[1], frame_size, "Frame size should be as specified.")
        self.assertTrue(np.array_equal(frames[0, :len(short_audio)], short_audio),
                        "Initial part of the frame should match the short audio.")
        self.assertTrue(np.all(frames[0, len(short_audio):] == 0),
                        "Padded part of the frame should be zeros.")

    def test_frame_audio_exact_fit(self):
        """Test framing when audio length is an exact multiple of hop_size + frame_size."""
        frame_size = 100
        hop_size = 50
        # Create audio that fits perfectly: L = num_hops * hop_size + frame_size
        # e.g., 2 hops: L = 2 * 50 + 100 = 200
        # This results in num_hops + 1 frames. So, 3 frames.
        # (0, 100), (50, 150), (100, 200)
        num_frames_target = 3
        exact_len_audio = np.arange((num_frames_target - 1) * hop_size + frame_size) # length = 2*50+100 = 200

        frames = frame_audio(exact_len_audio, frame_size=frame_size, hop_size=hop_size)
        self.assertEqual(frames.shape[0], num_frames_target)
        self.assertEqual(frames.shape[1], frame_size)

    def test_frame_audio_non_mono_input(self):
        """Test that frame_audio raises error for non-1D input."""
        with self.assertRaises(ValueError):
            frame_audio(self.test_audio_stereo, frame_size=1024, hop_size=512)


if __name__ == '__main__':
    unittest.main(verbosity=2)
