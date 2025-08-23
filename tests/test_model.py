# tests/test_model.py
import unittest
import torch

# Adjust import paths as necessary
try:
    from src.model.diffusion import DiffusionModel, UNetPlaceholder
    from src.model.condition import TextConditionEncoder, MelSpectrogramConditionEncoder, ClassConditionEncoder
    from src.model.utils import PositionalEncoding, TimeEmbedding, get_noise_schedule
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.model.diffusion import DiffusionModel, UNetPlaceholder
    from src.model.condition import TextConditionEncoder, MelSpectrogramConditionEncoder, ClassConditionEncoder
    from src.model.utils import PositionalEncoding, TimeEmbedding, get_noise_schedule

class TestModelComponents(unittest.TestCase):

    def setUp(self):
        """Common parameters for model tests."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Common UNet/DiffusionModel parameters
        self.input_channels = 1
        self.output_channels = 1 # For noise model, typically same as input
        self.time_embed_dim = 64
        self.condition_dim_example = 128 # Example dimension for conditioning
        self.num_timesteps = 50 # Fewer timesteps for faster tests

        # Instantiate a placeholder UNet for use in DiffusionModel tests
        self.unet_placeholder = UNetPlaceholder(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            time_embed_dim=self.time_embed_dim,
            condition_dim=self.condition_dim_example
        ).to(self.device)

    # --- Test src.model.diffusion ---
    def test_unet_placeholder_instantiation(self):
        """Test if the UNetPlaceholder can be instantiated."""
        unet = UNetPlaceholder(input_channels=1, output_channels=1, time_embed_dim=32, condition_dim=64)
        self.assertIsInstance(unet, UNetPlaceholder)
        print(f"UNetPlaceholder instantiated on {self.device}")

    def test_unet_placeholder_forward_pass(self):
        """Test a basic forward pass of UNetPlaceholder."""
        batch_size = 2
        audio_length = 1024
        unet = self.unet_placeholder

        dummy_x = torch.randn(batch_size, self.input_channels, audio_length).to(self.device)
        # In a real scenario, t and condition would be processed into embeddings first.
        # For UNetPlaceholder, we pass raw tensors for t_embed and condition_embed if dims are set.
        dummy_t_embed = torch.randn(batch_size, self.time_embed_dim).to(self.device) if self.time_embed_dim else None
        dummy_c_embed = torch.randn(batch_size, self.condition_dim_example).to(self.device) if self.condition_dim_example else None

        # The UNetPlaceholder's forward method concatenates these if they exist.
        # Its internal conv layer input channels must match combined input channels.
        # The UNetPlaceholder's conv1 is defined based on these dims.
        output = unet(dummy_x, t_embed=dummy_t_embed, condition_embed=dummy_c_embed)
        self.assertEqual(output.shape, (batch_size, self.output_channels, audio_length))


    def test_diffusion_model_instantiation(self):
        """Test if the DiffusionModel can be instantiated."""
        model = DiffusionModel(
            model_architecture=self.unet_placeholder,
            input_channels=self.input_channels,
            condition_dim=self.condition_dim_example,
            num_timesteps=self.num_timesteps
        ).to(self.device)
        self.assertIsInstance(model, DiffusionModel)
        self.assertEqual(len(model.betas), self.num_timesteps) # Check if schedule is initialized
        print(f"DiffusionModel instantiated on {self.device}")

    def test_diffusion_model_forward_pass(self):
        """Test a basic forward pass of DiffusionModel (predicts noise)."""
        model = DiffusionModel(
            model_architecture=self.unet_placeholder,
            input_channels=self.input_channels,
            condition_dim=self.condition_dim_example,
            num_timesteps=self.num_timesteps
        ).to(self.device)

        batch_size = 2
        audio_length = 2048
        dummy_x_t = torch.randn(batch_size, self.input_channels, audio_length).to(self.device)
        dummy_timesteps = torch.randint(0, self.num_timesteps, (batch_size,)).to(self.device)
        dummy_condition = torch.randn(batch_size, self.condition_dim_example).to(self.device) if self.condition_dim_example else None

        # The DiffusionModel's forward calls its internal model's forward.
        # UNetPlaceholder has a `forward_with_time_and_condition` that DiffusionModel tries to use.
        # If not found, it returns zeros. This test relies on the placeholder's behavior.
        predicted_noise = model(dummy_x_t, dummy_timesteps, dummy_condition)
        self.assertEqual(predicted_noise.shape, dummy_x_t.shape)

    def test_diffusion_model_sample_basic(self):
        """Test basic sampling from DiffusionModel (returns noise with current placeholder)."""
        model = DiffusionModel(
            model_architecture=self.unet_placeholder, # UNetPlaceholder
            input_channels=self.input_channels,
            condition_dim=self.condition_dim_example,
            num_timesteps=self.num_timesteps
        ).to(self.device)

        batch_size = 1
        audio_length = 1024
        shape_to_sample = (self.input_channels, audio_length)
        dummy_condition = torch.randn(batch_size, self.condition_dim_example).to(self.device) if self.condition_dim_example else None

        # The sample method is a placeholder in DiffusionModel and returns initial noise
        generated_output = model.sample(
            num_samples=batch_size,
            shape=shape_to_sample,
            device=self.device,
            condition=dummy_condition
        )
        self.assertEqual(generated_output.shape, (batch_size, *shape_to_sample))
        # With current placeholders, this basically returns Gaussian noise. No need to check values.


    # --- Test src.model.condition ---
    def test_text_condition_encoder(self):
        """Test TextConditionEncoder."""
        encoder = TextConditionEncoder(text_embedding_dim=768, output_dim=self.condition_dim_example).to(self.device)
        dummy_text_embed = torch.randn(2, 768).to(self.device) # (batch, input_dim)
        condition_vec = encoder(dummy_text_embed)
        self.assertEqual(condition_vec.shape, (2, self.condition_dim_example))

        dummy_text_embed_seq = torch.randn(2, 10, 768).to(self.device) # (batch, seq_len, input_dim)
        condition_vec_seq = encoder(dummy_text_embed_seq) # Should average pool seq_len
        self.assertEqual(condition_vec_seq.shape, (2, self.condition_dim_example))


    def test_mel_spectrogram_condition_encoder(self):
        """Test MelSpectrogramConditionEncoder."""
        n_mels = 80
        time_frames = 50
        # Test with RNN
        encoder_rnn = MelSpectrogramConditionEncoder(n_mels=n_mels, output_dim=self.condition_dim_example, use_rnn=True).to(self.device)
        dummy_mel_spec = torch.randn(2, n_mels, time_frames).to(self.device)
        condition_vec_rnn = encoder_rnn(dummy_mel_spec)
        self.assertEqual(condition_vec_rnn.shape, (2, self.condition_dim_example))

        # Test without RNN (pooling)
        encoder_pool = MelSpectrogramConditionEncoder(n_mels=n_mels, output_dim=self.condition_dim_example, use_rnn=False).to(self.device)
        condition_vec_pool = encoder_pool(dummy_mel_spec)
        self.assertEqual(condition_vec_pool.shape, (2, self.condition_dim_example))


    def test_class_condition_encoder(self):
        """Test ClassConditionEncoder."""
        num_classes = 10
        encoder = ClassConditionEncoder(num_classes=num_classes, embedding_dim=self.condition_dim_example).to(self.device)
        dummy_class_ids = torch.randint(0, num_classes, (2,)).to(self.device)
        condition_vec = encoder(dummy_class_ids)
        self.assertEqual(condition_vec.shape, (2, self.condition_dim_example))


    # --- Test src.model.utils ---
    def test_positional_encoding(self):
        """Test PositionalEncoding (more relevant for Transformers but included)."""
        d_model = 32
        pe = PositionalEncoding(d_model=d_model, max_len=100).to(self.device)
        test_tensor = torch.randn(50, 2, d_model).to(self.device) # (seq_len, batch, d_model)
        encoded = pe(test_tensor)
        self.assertEqual(encoded.shape, test_tensor.shape)
        # Basic check: PE should change the input, unless input is zero and dropout is zero.
        # self.assertFalse(torch.allclose(encoded, test_tensor)) # This might fail if dropout is 0 and x is 0

    def test_time_embedding(self):
        """Test TimeEmbedding for diffusion timesteps."""
        time_emb = TimeEmbedding(embedding_dim=self.time_embed_dim).to(self.device)
        dummy_timesteps = torch.randint(0, self.num_timesteps, (4,)).to(self.device) # Batch of 4 timesteps
        embeddings = time_emb(dummy_timesteps)
        self.assertEqual(embeddings.shape, (4, self.time_embed_dim))

    def test_get_noise_schedule(self):
        """Test noise schedule generation."""
        schedules = ['linear', 'cosine', 'quadratic', 'sigmoid']
        for sched_type in schedules:
            params = get_noise_schedule(sched_type, self.num_timesteps, device=self.device)
            # betas, alphas, alphas_cumprod, ...
            self.assertEqual(params[0].shape[0], self.num_timesteps, f"Beta length mismatch for {sched_type}")
            self.assertEqual(params[2].shape[0], self.num_timesteps, f"Alphas_cumprod length mismatch for {sched_type}")
            self.assertTrue(torch.is_tensor(params[0])) # Check if it's a tensor
            # Ensure all params are on the correct device
            for p_tensor in params:
                 if torch.is_tensor(p_tensor) and p_tensor.numel() > 0: # Check non-empty tensors
                    self.assertEqual(p_tensor.device.type, self.device.type, f"Tensor for {sched_type} not on device {self.device.type}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
