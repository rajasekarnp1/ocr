# src/postprocessing.py
"""
Handles audio postprocessing tasks like denoising, format conversion, etc.
(Potentially less critical for an initial diffusion model structure,
but good to have as a placeholder)
"""
import numpy as np
# import librosa # if advanced STFT/ISTFT based denoising is needed
# from scipy.signal import wiener # Example for Wiener filter

def simple_noise_gate(audio_data, threshold_db=-50, attack_time=0.01, release_time=0.1, sr=44100):
    """
    A very simple noise gate.
    Note: This is a basic implementation. For serious denoising,
    consider dedicated libraries or more sophisticated algorithms.

    Args:
        audio_data (np.ndarray): Input audio data.
        threshold_db (float): Threshold in dB. Audio below this will be attenuated.
        attack_time (float): Time in seconds for the gate to open.
        release_time (float): Time in seconds for the gate to close.
        sr (int): Sampling rate.

    Returns:
        np.ndarray: Audio data with noise gate applied.
    """
    if audio_data.size == 0:
        return audio_data

    threshold = 10**(threshold_db / 20)

    # Calculate envelope (RMS based)
    # A more sophisticated envelope follower would be better
    frame_length = int(0.02 * sr) # 20ms frames
    hop_length = int(0.01 * sr)   # 10ms hop

    if len(audio_data) < frame_length: # Handle very short audio
        rms = np.sqrt(np.mean(audio_data**2))
        if rms < threshold:
            return np.zeros_like(audio_data)
        else:
            return audio_data

    # Using librosa's framing for RMS calculation for simplicity here
    # This is not a real-time envelope follower for a proper gate
    # but gives a general idea.
    # A proper implementation would iterate sample by sample or use filter designs.

    # For simplicity, this example will apply a hard gate based on overall RMS.
    # This is NOT a proper time-varying noise gate.
    rms_overall = np.sqrt(np.mean(audio_data**2))

    if rms_overall < threshold:
        # Attenuate if overall RMS is below threshold (very crude)
        # A real gate works on segments and has attack/release.
        # This is more like a "silence if quiet overall"
        # return audio_data * 0.1 # Example attenuation
        print("Warning: This simple_noise_gate is very basic and acts globally.")
        print("For proper gating, a more advanced implementation is needed.")
        return np.zeros_like(audio_data) # Simple hard gate for this example
    else:
        return audio_data

def fade_in_out(audio_data, fade_duration_ms=10, sr=44100):
    """
    Applies a short fade-in and fade-out to the audio.

    Args:
        audio_data (np.ndarray): Input audio.
        fade_duration_ms (int): Duration of the fade in milliseconds.
        sr (int): Sampling rate.

    Returns:
        np.ndarray: Audio with fades applied.
    """
    if audio_data.size == 0:
        return audio_data

    fade_samples = int((fade_duration_ms / 1000.0) * sr)

    if fade_samples == 0:
        return audio_data # No fade if duration is too short

    if len(audio_data) < fade_samples * 2: # If audio is shorter than total fade time
        # Apply shorter fades or just return
        fade_samples = len(audio_data) // 2
        if fade_samples == 0:
            return audio_data


    # Fade in
    fade_in_curve = np.linspace(0., 1., fade_samples)
    audio_data[:fade_samples] *= fade_in_curve

    # Fade out
    fade_out_curve = np.linspace(1., 0., fade_samples)
    audio_data[-fade_samples:] *= fade_out_curve

    return audio_data

if __name__ == '__main__':
    # Example Usage
    import numpy as np
    from audio_io import write_audio # Assuming audio_io.py is in the same directory or PYTHONPATH

    SAMPLE_RATE = 44100
    DURATION = 2
    FREQUENCY = 440
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)
    test_audio_clean = 0.5 * np.sin(2 * np.pi * FREQUENCY * t)

    # Add some noise for the noise gate example (very artificial noise)
    noise = (np.random.rand(len(test_audio_clean)) - 0.5) * 0.01
    test_audio_noisy = test_audio_clean + noise

    # 1. Test Noise Gate (basic version)
    # Note: The current simple_noise_gate is very rudimentary.
    # For real applications, you'd use something like Wiener filter or spectral gating.
    gated_audio = simple_noise_gate(test_audio_noisy, threshold_db=-40, sr=SAMPLE_RATE)
    # write_audio("gated_example.wav", gated_audio, SAMPLE_RATE)
    print(f"Gated audio length: {len(gated_audio)}")

    # 2. Test Fade In/Out
    faded_audio = fade_in_out(test_audio_clean.copy(), fade_duration_ms=50, sr=SAMPLE_RATE)
    # write_audio("faded_example.wav", faded_audio, SAMPLE_RATE)
    print(f"Faded audio sample (start): {faded_audio[:5]}")
    print(f"Faded audio sample (end): {faded_audio[-5:]}")

    # Test with very short audio
    short_audio = test_audio_clean[:100]
    faded_short_audio = fade_in_out(short_audio.copy(), fade_duration_ms=50, sr=SAMPLE_RATE)
    print(f"Faded short audio length: {len(faded_short_audio)}")
    # write_audio("faded_short_example.wav", faded_short_audio, SAMPLE_RATE)

    # Test with silent audio
    silent_audio = np.zeros(SAMPLE_RATE)
    faded_silent = fade_in_out(silent_audio.copy())
    print(f"Faded silent audio: {faded_silent[:5]}")
