import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class ConditioningEncoder(nn.Module):
    """
    Encodes the low-resolution audio condition into feature maps or an embedding.
    This is a placeholder and can be made more sophisticated.
    Example: A simple CNN to extract features from the low-resolution audio.
    """
    def __init__(self,
                 in_channels: int = 1,
                 base_channels: int = 32,
                 output_channels: int = None, # If None, defaults to base_channels * (stride chosen for last conv layer)
                 num_layers: int = 4,
                 stride: int = 2): # Stride for downsampling/feature extraction
        super().__init__()

        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()

        current_channels = in_channels

        for i in range(num_layers):
            # Determine output channels for this layer
            if i < num_layers - 1:
                # Intermediate layers scale up base_channels by stride factor progressively
                # Example: base=32, stride=2 -> 32, 64, 128 for first 3 layers if num_layers is high enough
                # This logic might need adjustment based on desired channel progression
                out_ch_layer = base_channels * (stride**i)
            else: # Last layer
                out_ch_layer = output_channels if output_channels is not None else current_channels # Or base_channels * (stride**(i))
                if out_ch_layer is None: # Ensure it's set if output_channels was None
                    out_ch_layer = base_channels * (stride**(i)) if i > 0 else base_channels


            # Determine stride for this layer
            # Stride only for intermediate layers if we want final layer to preserve length for feature map output
            # Or, always stride if we want maximal downsampling.
            # The example test implies last layer has stride 1.
            layer_stride = stride if i < num_layers - 1 else 1

            conv = nn.Conv1d(
                current_channels,
                out_ch_layer,
                kernel_size=3,
                stride=layer_stride,
                padding=1
            )
            self.conv_layers.append(conv)

            # GroupNorm: ensure num_groups is valid and <= num_channels
            num_groups = min(32, out_ch_layer) if out_ch_layer > 0 else 1
            if out_ch_layer > 0 and out_ch_layer % num_groups != 0: # Find a valid num_groups
                # Fallback: try smaller common factors or 1 if prime/small
                if out_ch_layer % 16 == 0: num_groups = 16
                elif out_ch_layer % 8 == 0: num_groups = 8
                elif out_ch_layer % 4 == 0: num_groups = 4
                elif out_ch_layer % 2 == 0: num_groups = 2
                else: num_groups = 1 # Smallest possible group

            norm = nn.GroupNorm(num_groups if out_ch_layer > 0 else 1, out_ch_layer if out_ch_layer > 0 else 1)

            self.conv_layers.append(norm)
            self.conv_layers.append(nn.SiLU())
            current_channels = out_ch_layer

        self.final_out_channels = current_channels # Store the actual output channels of the last layer
        logger.info(f"ConditioningEncoder initialized. In: {in_channels}, Base: {base_channels}, Actual Out: {self.final_out_channels}, Layers: {num_layers}, Strides per layer (approx): {stride} for first {num_layers-1}, then 1.")


    def forward(self, low_res_audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            low_res_audio (torch.Tensor): The low-resolution audio input.
                                         Shape: (batch_size, in_channels, length_low_res)
        Returns:
            torch.Tensor: Conditioning feature maps.
                          Shape: (batch_size, self.final_out_channels, feature_length)
        """
        h = low_res_audio
        for layer in self.conv_layers:
            h = layer(h)
        return h


class FiLMGenerator(nn.Module):
    """
    Generates scale (gamma) and shift (beta) parameters for FiLM layers
    from a conditioning embedding.
    """
    def __init__(self, cond_embedding_dim: int, target_channels: int):
        super().__init__()
        self.fc = nn.Linear(cond_embedding_dim, target_channels * 2) # *2 for gamma and beta
        self.target_channels = target_channels
        logger.info(f"FiLMGenerator initialized. Cond dim: {cond_embedding_dim}, Target channels for FiLM: {target_channels}")

    def forward(self, cond_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cond_embedding (torch.Tensor): Conditioning embedding.
                                         Shape: (batch_size, cond_embedding_dim)
        Returns:
            tuple[torch.Tensor, torch.Tensor]: gamma, beta
                                              Shape: (batch_size, target_channels)
        """
        params = self.fc(cond_embedding)
        gamma = params[:, :self.target_channels]
        beta = params[:, self.target_channels:]
        return gamma, beta


if __name__ == '__main__':
    logger.info("Starting model condition.py example usage...")

    batch_size = 2
    lr_audio_channels = 1
    lr_audio_length = 1024

    cond_base_ch = 16
    # Let's define target output channels for the encoder explicitly
    # The encoder logic was a bit ambiguous for `output_channels=None`
    # If num_layers=3, stride=2.
    # Layer 0: in=1, out=16, stride=2. L_out = 1024/2 = 512
    # Layer 1: in=16, out=16*2=32, stride=2. L_out = 512/2 = 256
    # Layer 2: in=32, out=TARGET_OUT_CH (e.g. 64), stride=1. L_out = 256
    target_cond_encoder_out_channels = 64

    simple_conditioner = ConditioningEncoder(
        in_channels=lr_audio_channels,
        base_channels=cond_base_ch,
        output_channels=target_cond_encoder_out_channels, # Explicitly set output channels of the final conv layer
        num_layers=3,
        stride=2
    )
    logger.info(f"SimpleConditioner model structure: {simple_conditioner}")
    num_params_cond = sum(p.numel() for p in simple_conditioner.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters in SimpleConditioner: {num_params_cond}")

    dummy_lr_audio = torch.randn(batch_size, lr_audio_channels, lr_audio_length)
    logger.info(f"Dummy low-res audio input shape: {dummy_lr_audio.shape}")

    try:
        conditioning_features = simple_conditioner(dummy_lr_audio)
        logger.info(f"Conditioning features output shape: {conditioning_features.shape}")

        # Calculate expected output length:
        # L_in = 1024
        # Layer 0 (Conv1d): stride=2, padding=1, kernel=3. L_out0 = floor((1024 + 2*1 - 1*(3-1) - 1)/2 + 1) = floor((1024)/2) = 512
        # Layer 1 (Conv1d): stride=2, padding=1, kernel=3. L_out1 = floor((512 + 2*1 - 1*(3-1) - 1)/2 + 1) = floor((512)/2) = 256
        # Layer 2 (Conv1d): stride=1, padding=1, kernel=3. L_out2 = floor((256 + 2*1 - 1*(3-1) - 1)/1 + 1) = floor((256)/1) = 256
        expected_feature_length = lr_audio_length // 4 # Because 2 layers had stride 2

        assert conditioning_features.shape == (batch_size, target_cond_encoder_out_channels, expected_feature_length), \
            f"Conditioning features shape mismatch. Expected {(batch_size, target_cond_encoder_out_channels, expected_feature_length)}, got {conditioning_features.shape}"
        logger.info("ConditioningEncoder forward pass successful.")

        # Test FiLMGenerator
        # Assume features are pooled to an embedding. Use actual output channels of encoder.
        pooled_features_dim = simple_conditioner.final_out_channels
        simulated_embedding = torch.mean(conditioning_features, dim=2) # (batch_size, pooled_features_dim)
        logger.info(f"Simulated pooled embedding shape: {simulated_embedding.shape}")

        film_target_channels = 128
        film_generator = FiLMGenerator(
            cond_embedding_dim=pooled_features_dim,
            target_channels=film_target_channels
        )
        logger.info(f"FiLMGenerator model structure: {film_generator}")

        gamma, beta = film_generator(simulated_embedding)
        logger.info(f"FiLM gamma shape: {gamma.shape}, beta shape: {beta.shape}")
        assert gamma.shape == (batch_size, film_target_channels), "FiLM gamma shape mismatch."
        assert beta.shape == (batch_size, film_target_channels), "FiLM beta shape mismatch."
        logger.info("FiLMGenerator forward pass successful.")

    except ImportError:
        logger.error("Torch import failed. Cannot run condition.py example. This is expected if torch is not installed.")
    except Exception as e:
        logger.error(f"Error during condition.py example usage: {e}", exc_info=True)

    logger.info("model condition.py example usage finished.")
