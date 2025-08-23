# src/training/logger.py
"""
Handles logging of training metrics, model checkpoints, and generated audio samples.
Supports TensorBoard and potentially other logging backends like Weights & Biases.
"""
import os
import json
import torch
from torch.utils.tensorboard import SummaryWriter
import soundfile as sf
import datetime

class TrainingLogger:
    def __init__(self, log_dir="logs", config_to_save=None, project_name="AudioDiffusion", use_tensorboard=True):
        """
        Initializes the TrainingLogger.

        Args:
            log_dir (str): Directory to save logs and checkpoints.
            config_to_save (dict, optional): Configuration dictionary to save.
            project_name (str): Name of the project/experiment.
            use_tensorboard (bool): Whether to use TensorBoard for logging.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.base_log_dir = log_dir
        self.experiment_dir = os.path.join(log_dir, f"{project_name}_{timestamp}")
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.audio_log_dir = os.path.join(self.experiment_dir, "audio_samples")

        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.audio_log_dir, exist_ok=True)

        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=self.experiment_dir)
        else:
            self.writer = None
            print("TensorBoard logging disabled. Metrics will be printed to console/stored locally if implemented.")

        if config_to_save:
            self.save_config(config_to_save)

        print(f"TrainingLogger initialized. Logs will be saved in: {self.experiment_dir}")

    def save_config(self, config_dict):
        """Saves the configuration dictionary as a JSON file."""
        config_path = os.path.join(self.experiment_dir, "config.json")
        try:
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=4)
            print(f"Configuration saved to {config_path}")
        except TypeError as e:
            print(f"Error saving config (likely due to non-serializable objects): {e}")
            print("Attempting to save a string representation of the config.")
            with open(config_path, 'w') as f:
                f.write(str(config_dict))


    def log_scalar(self, tag, value, step):
        """Logs a scalar value (e.g., loss, learning rate)."""
        if self.writer:
            self.writer.add_scalar(tag, value, step)
        else:
            print(f"Step {step} | {tag}: {value}") # Fallback if no TensorBoard

    def log_audio(self, tag, audio_data, step_or_epoch, sample_rate=44100, is_epoch_log=True):
        """
        Logs audio data to TensorBoard and saves it as a .wav file.

        Args:
            tag (str): Name for the audio log.
            audio_data (torch.Tensor or np.ndarray): Audio data (1D or 2D for mono/stereo).
                                                     Should be in [-1, 1] range.
            step_or_epoch (int): Current training step or epoch number.
            sample_rate (int): Sampling rate of the audio.
            is_epoch_log (bool): If True, filenames include 'epoch', else 'step'.
        """
        if audio_data is None:
            print(f"Warning: Audio data for tag '{tag}' is None. Skipping logging.")
            return

        # Ensure audio_data is suitable for TensorBoard and soundfile
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.detach().cpu().numpy()

        if audio_data.ndim == 2 and audio_data.shape[0] == 1: # (1, N) -> (N,)
            audio_data_tb = audio_data
        elif audio_data.ndim == 1: # (N,) -> (1, N) for TensorBoard
             audio_data_tb = audio_data.reshape(1, -1)
        else: # Potentially (N, C) or other formats not directly supported by add_audio
            print(f"Warning: Audio data for tag '{tag}' has unexpected shape {audio_data.shape}. Attempting to log first channel or as is.")
            if audio_data.ndim > 1 and audio_data.shape[0] > 1 and audio_data.shape[1] > 0: # e.g. (N, C)
                 audio_data_tb = audio_data[:,0].reshape(1,-1) # Take first channel, make (1,N)
            else:
                 audio_data_tb = audio_data # Log as is, may error or not display well

        # Log to TensorBoard
        if self.writer:
            try:
                self.writer.add_audio(tag, audio_data_tb, global_step=step_or_epoch, sample_rate=sample_rate)
            except Exception as e:
                print(f"Error logging audio to TensorBoard for tag '{tag}': {e}")
                print("Audio data shape was:", audio_data_tb.shape)


        # Save as .wav file
        prefix = "epoch" if is_epoch_log else "step"
        filename = f"{tag.replace('/', '_')}_{prefix}_{step_or_epoch}.wav"
        filepath = os.path.join(self.audio_log_dir, filename)
        try:
            # soundfile expects (frames, channels) or (frames,)
            # If audio_data is (channels, frames) from torch, transpose it.
            if audio_data.ndim == 2 and audio_data.shape[0] < audio_data.shape[1]: # Likely (C, N)
                audio_data_sf = audio_data.T
            else: # Already (N,) or (N, C)
                audio_data_sf = audio_data

            sf.write(filepath, audio_data_sf, sample_rate)
        except Exception as e:
            print(f"Error saving audio file {filepath}: {e}")
            print("Original audio_data shape:", audio_data.shape, "Attempted sf shape:", audio_data_sf.shape)


    def save_checkpoint(self, model, optimizer, epoch, step, config, filename_prefix="model"):
        """
        Saves a model and optimizer checkpoint.

        Args:
            model (torch.nn.Module): Model to save.
            optimizer (torch.optim.Optimizer): Optimizer state to save.
            epoch (int): Current epoch.
            step (int): Current global training step.
            config (dict): Training configuration (for reference).
            filename_prefix (str): Prefix for the checkpoint filename.
        """
        checkpoint_name = f"{filename_prefix}_epoch_{epoch+1}_step_{step}.pth"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        state = {
            'epoch': epoch,
            'global_step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config # Storing config with checkpoint can be useful
        }

        torch.save(state, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        # Optionally, save a "latest" checkpoint symlink or copy
        latest_path = os.path.join(self.checkpoint_dir, f"{filename_prefix}_latest.pth")
        torch.save(state, latest_path) # Overwrites previous "latest"


    def load_checkpoint(self, model, optimizer, filepath, device):
        """
        Loads a model and optimizer checkpoint.

        Args:
            model (torch.nn.Module): Model to load state into.
            optimizer (torch.optim.Optimizer): Optimizer to load state into.
            filepath (str): Path to the checkpoint file.
            device (torch.device): Device to map loaded tensors to.

        Returns:
            tuple: (epoch, global_step, config) loaded from checkpoint.
                   Returns (0, 0, None) if checkpoint not found.
        """
        if not os.path.exists(filepath):
            print(f"Checkpoint file not found: {filepath}")
            return 0, 0, None

        checkpoint = torch.load(filepath, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        config_from_ckpt = checkpoint.get('config', None)

        print(f"Checkpoint loaded from {filepath}. Resuming from epoch {epoch+1}, global_step {global_step}.")
        return epoch, global_step, config_from_ckpt

    def close(self):
        """Closes the TensorBoard writer."""
        if self.writer:
            self.writer.close()
        print("TrainingLogger closed.")

if __name__ == '__main__':
    print("--- TrainingLogger Example ---")

    # Dummy components for testing
    dummy_model = torch.nn.Linear(10, 2) # A very simple model
    dummy_optimizer = torch.optim.Adam(dummy_model.parameters())
    dummy_config = {"lr": 0.001, "batch_size": 32, "arch": "SimpleLinear"}

    # 1. Initialize Logger
    test_log_dir = "logger_test_logs"
    logger = TrainingLogger(log_dir=test_log_dir, config_to_save=dummy_config, project_name="LoggerTest")

    # 2. Log Scalars
    print("\nLogging scalars...")
    for i in range(5):
        logger.log_scalar("ExampleLoss/train", 1.0 / (i + 1), i)
        logger.log_scalar("ExampleMetric/accuracy", 0.5 + i * 0.1, i)

    # 3. Log Audio
    print("\nLogging audio...")
    sample_rate = 16000
    # Mono audio
    dummy_audio_mono = torch.sin(torch.arange(0, sample_rate * 1.0) * 0.1) # 1 sec sine wave
    logger.log_audio("GeneratedSamples/mono_test", dummy_audio_mono, step_or_epoch=0, sample_rate=sample_rate)

    # Stereo audio (for file saving, tensorboard takes (1, N) or (N,))
    # For soundfile, it can be (N, C) or (N,). For TB, it should be (1, N) effectively.
    dummy_audio_stereo_torch = torch.stack([dummy_audio_mono * 0.8, dummy_audio_mono * 0.6], dim=0) # (2, N)
    # Tensorboard add_audio expects (1, L) or (L,)
    # So we log first channel to Tensorboard, but save the stereo .wav
    logger.log_audio("GeneratedSamples/stereo_test_tb_ch1", dummy_audio_stereo_torch[0,:], step_or_epoch=0, sample_rate=sample_rate)
    # To save the stereo file correctly with the logger, the logger's audio saving part needs to handle (C,N) or (N,C)
    # The current logger.log_audio saves `audio_data` which for stereo_torch is (2,N)
    # soundfile can handle (N,C) so it should be transposed if (C,N)
    # Let's make a specific call to test the .wav saving of stereo
    stereo_filepath = os.path.join(logger.audio_log_dir, "stereo_manual_save_test.wav")
    sf.write(stereo_filepath, dummy_audio_stereo_torch.T.numpy(), sample_rate) # Transpose (2,N) to (N,2)
    print(f"Manually saved stereo audio to {stereo_filepath} for verification.")


    # 4. Save Checkpoint
    print("\nSaving checkpoint...")
    logger.save_checkpoint(dummy_model, dummy_optimizer, epoch=0, step=100, config=dummy_config, filename_prefix="test_model")

    # 5. Load Checkpoint (example)
    print("\nLoading checkpoint (example)...")
    # Create new instances to simulate loading into a fresh setup
    new_model = torch.nn.Linear(10, 2)
    new_optimizer = torch.optim.Adam(new_model.parameters())
    checkpoint_file_path = os.path.join(logger.checkpoint_dir, "test_model_epoch_1_step_100.pth")
    if not os.path.exists(checkpoint_file_path): # Fallback to latest if specific one not found (e.g. due to +1 in epoch naming)
        checkpoint_file_path = os.path.join(logger.checkpoint_dir, "test_model_latest.pth")

    loaded_epoch, loaded_step, loaded_config = logger.load_checkpoint(
        new_model, new_optimizer, checkpoint_file_path, device=torch.device('cpu')
    )
    if loaded_config:
        print(f"Loaded config from checkpoint: {loaded_config.get('arch', 'N/A')}")
    else:
        print("Could not load config from checkpoint (or checkpoint not found).")


    # 6. Close Logger
    logger.close()
    print(f"\nTest logs, checkpoints, and audio saved in: {logger.experiment_dir}")
    print("Please inspect the directory and TensorBoard (if run with `tensorboard --logdir .`)")

    # Rudimentary cleanup (optional, be careful with rmdir on real logs)
    # import shutil
    # try:
    #     shutil.rmtree(test_log_dir)
    #     print(f"\nCleaned up test directory: {test_log_dir}")
    # except OSError as e:
    #     print(f"Error cleaning up test directory {test_log_dir}: {e}")
    #     print("You may need to remove it manually.")

    print("\nTrainingLogger example finished.")
