import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
# Ensure .utils and .condition can be imported relative to src/model/
# This might require python -m src.model.diffusion if running directly,
# or proper PYTHONPATH setup / project installation.
# For subtask execution, assume relative imports work if files are in place.
from .utils import TimeEmbedding
# from .condition import ConditioningEncoder, FiLMGenerator # FiLMGenerator not used directly from here now

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class ResidualBlock(nn.Module):
    """
    Residual block with time embedding and FiLM conditioning.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_emb_dim, dropout_rate=0.1, use_attention=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Helper to determine num_groups for GroupNorm
        def get_num_groups(ch):
            if ch == 0: return 1 # Should not happen if used correctly
            if ch < 32 :
                # Find largest factor <= ch, or 1. Common factors: 16, 8, 4, 2, 1
                for factor in [16,8,4,2,1]:
                    if ch % factor == 0: return ch // factor # Use num_groups = ch / factor
                return 1 # Fallback if no small common factor, effectively LayerNorm per channel
            return 32


        self.norm1 = nn.GroupNorm(get_num_groups(in_channels), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)

        if cond_emb_dim is not None:
            self.film_generator = nn.Linear(cond_emb_dim, out_channels * 2)
        else:
            self.film_generator = None

        self.norm2 = nn.GroupNorm(get_num_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.use_attention = use_attention
        if self.use_attention:
            self.attention = AttentionBlock(out_channels)
        logger.debug(f"ResBlock: in={in_channels}, out={out_channels}, t_emb={time_emb_dim}, c_emb={cond_emb_dim}, att={use_attention}")


    def forward(self, x, t_emb, cond_embedding=None): # cond_embedding can be None
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        time_cond = self.time_emb_proj(F.silu(t_emb))
        h = h + time_cond.unsqueeze(-1)

        if self.film_generator and cond_embedding is not None:
            film_params = self.film_generator(cond_embedding)
            gamma = film_params[:, :self.out_channels].unsqueeze(-1)
            beta = film_params[:, self.out_channels:].unsqueeze(-1)
            h = gamma * h + beta

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        x_shortcut = self.shortcut(x)
        out = h + x_shortcut

        if self.use_attention:
            out = self.attention(out)

        return out

class AttentionBlock(nn.Module):
    """ Simple self-attention block. """
    def __init__(self, channels):
        super().__init__()
        def get_num_groups(ch): # Duplicated helper, consider moving to utils if more common
            if ch == 0: return 1
            if ch < 32:
                for factor in [16,8,4,2,1]:
                    if ch % factor == 0: return ch // factor
                return 1
            return 32
        self.norm = nn.GroupNorm(get_num_groups(channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        h_ = self.norm(x)
        q, k, v = self.qkv(h_).chunk(3, dim=1)
        b, c, l = q.shape
        q = q.permute(0, 2, 1)
        # k remains (B, C, L)
        v = v.permute(0, 2, 1)
        w = torch.bmm(q, k) * (c ** -0.5) # QK^T / sqrt(d_k)
        w = F.softmax(w, dim=-1)
        h_ = torch.bmm(w, v) # (B, L, C)
        h_ = h_.permute(0, 2, 1) # (B, C, L)
        return x + self.proj_out(h_)

class DownsampleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)

class UpsampleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)

class DiffusionUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        model_channels: int = 64,
        out_channels: int = 1,
        num_residual_blocks: int = 2,
        channel_mult: tuple[int, ...] = (1, 2, 2, 2), # Example: (1,2,2,2) for 3 downsample layers
        time_emb_dim: int = 256,
        cond_emb_dim: int = None, # Dimension of the global conditioning embedding. If None, unconditional.
        dropout_rate: float = 0.1,
        use_attention_at_resolution: tuple[int, ...] = (1,2) # Resolution indices (0-indexed from input)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_residual_blocks = num_residual_blocks
        self.channel_mult = channel_mult
        self.time_emb_dim = time_emb_dim
        self.cond_emb_dim = cond_emb_dim

        try:
            self.time_embedding = TimeEmbedding(time_emb_dim)
            actual_time_emb_dim = time_emb_dim
        except ValueError as e:
            logger.warning(f"Adjusting time_emb_dim for TimeEmbedding from {time_emb_dim} to {time_emb_dim + (time_emb_dim % 2)} due to: {e}")
            actual_time_emb_dim = time_emb_dim + (time_emb_dim % 2) if time_emb_dim % 2 != 0 else time_emb_dim
            self.time_embedding = TimeEmbedding(actual_time_emb_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(actual_time_emb_dim, actual_time_emb_dim), nn.SiLU(),
            nn.Linear(actual_time_emb_dim, actual_time_emb_dim)
        )

        self.initial_conv = nn.Conv1d(in_channels, model_channels, kernel_size=3, padding=1)

        down_blocks_list = []
        current_channels = model_channels
        num_resolutions = len(channel_mult)

        logger.info("Constructing Downsampling Path...")
        for i in range(num_resolutions):
            res_out_channels = model_channels * channel_mult[i]
            logger.debug(f"  Down ResLevel {i}: Target channels {res_out_channels}")
            for j in range(num_residual_blocks):
                down_blocks_list.append(ResidualBlock(
                    current_channels, res_out_channels, actual_time_emb_dim, cond_emb_dim,
                    dropout_rate, use_attention=(i in use_attention_at_resolution)
                ))
                current_channels = res_out_channels
            if i != num_resolutions - 1: # Don't add downsample at the last (deepest) resolution
                logger.debug(f"    Downsampling from {current_channels} channels")
                down_blocks_list.append(DownsampleBlock(current_channels))
        self.down_blocks = nn.ModuleList(down_blocks_list)

        logger.info("Constructing Bottleneck...")
        self.bottleneck_res1 = ResidualBlock(current_channels, current_channels, actual_time_emb_dim, cond_emb_dim, dropout_rate, use_attention=True)
        self.bottleneck_res2 = ResidualBlock(current_channels, current_channels, actual_time_emb_dim, cond_emb_dim, dropout_rate, use_attention=False)
        logger.debug(f"  Bottleneck channels: {current_channels}")

        up_blocks_list = []
        logger.info("Constructing Upsampling Path...")
        for i in reversed(range(num_resolutions)):
            res_out_channels_target = model_channels * channel_mult[i] # Target for this resolution level
            logger.debug(f"  Up ResLevel {i} (reversed): Target output channels {res_out_channels_target}")

            for j in range(num_residual_blocks + 1): # +1 to account for concat with skip
                # Input channels for this ResBlock:
                # current_channels (from previous upsample layer or bottleneck)
                # + res_out_channels_target (from corresponding skip connection) ONLY for the first block in this level.
                block_in_channels = current_channels + res_out_channels_target if j == 0 else res_out_channels_target

                up_blocks_list.append(ResidualBlock(
                    block_in_channels,
                    res_out_channels_target,
                    actual_time_emb_dim, cond_emb_dim, dropout_rate,
                    use_attention=(i in use_attention_at_resolution)
                ))
                current_channels = res_out_channels_target # Output of this ResBlock is the new current_channels

            if i != 0: # Don't add upsample at the first resolution level (i.e., output level)
                logger.debug(f"    Upsampling from {current_channels} channels")
                up_blocks_list.append(UpsampleBlock(current_channels))
        self.up_blocks = nn.ModuleList(up_blocks_list)

        # Final projection layer
        # current_channels should be model_channels if channel_mult[0] is 1
        def get_num_groups_final(ch): # Helper
            if ch == 0: return 1
            if ch < 32:
                for factor in [16,8,4,2,1]:
                    if ch % factor == 0: return ch // factor
                return 1
            return 32

        self.final_norm = nn.GroupNorm(get_num_groups_final(current_channels), current_channels)
        self.final_conv = nn.Conv1d(current_channels, out_channels, kernel_size=3, padding=1)

        logger.info(f"DiffusionUNet initialized. Cond_emb_dim: {cond_emb_dim if cond_emb_dim else 'None (Unconditional)'}")


    def forward(self, x, time, cond_embedding=None):
        if self.cond_emb_dim is not None and cond_embedding is None:
            raise ValueError("Model configured for conditioning, but cond_embedding was not provided.")
        if self.cond_emb_dim is None and cond_embedding is not None:
            logger.warning("Model is unconditional, but cond_embedding was provided. It will be ignored.")
            cond_embedding = None # Ensure it's None if model is unconditional

        t_emb = self.time_embedding(time)
        t_emb = self.time_mlp(t_emb)

        h = self.initial_conv(x)
        skip_connections = []

        # Downsampling Path
        for block in self.down_blocks:
            if isinstance(block, ResidualBlock):
                h = block(h, t_emb, cond_embedding)
                skip_connections.append(h)
            else: # DownsampleBlock
                h = block(h)

        # Bottleneck
        h = self.bottleneck_res1(h, t_emb, cond_embedding)
        h = self.bottleneck_res2(h, t_emb, cond_embedding)

        # Upsampling Path
        # This loop structure needs to be carefully managed with skip_connections
        # and knowledge of whether a block is an UpsampleBlock or ResidualBlock.

        # Iterating through up_blocks, which contains both Upsample and Residual blocks sequentially for each level
        # For each resolution level (iterating i in reversed order):
        #   If not the first level (i.e. i != 0), current h is from a lower res. It needs upsampling.
        #   Then, concat with skip.
        #   Then, pass through (num_residual_blocks + 1) ResBlocks.

        # Let's try to reconstruct the path based on how up_blocks was built:
        # It was built as: [Res, Res, ..., Upsample(opt), Res, Res, ..., Upsample(opt), ...]
        # This means an UpsampleBlock prepares `h` for the *next* (higher-res) level's ResBlocks.
        # Skip connections need to be fed to the *first* ResBlock of each upsampling stage.

        up_block_idx = 0
        for i_level in reversed(range(len(self.channel_mult))):
            # Upsample if not the deepest level (which is handled by bottleneck output `h`)
            # The upsample block for level i (from bottom, i.e. i=0 is deepest in up path)
            # should make `h` have the channel count of `model_channels * channel_mult[i]` (reversed)
            if i_level != len(self.channel_mult) -1 : # If not the level that comes from bottleneck directly
                 # This assumes an upsample was the last block of the *previous* (lower-res) iteration
                 # This logic is getting complex and error-prone due to the flat list `self.up_blocks`
                 # A better structure would be ModuleLists per resolution level.
                 # For now, assume the current `h` is the output of an UpsampleBlock if needed.
                 pass

            for j_res_block in range(self.num_residual_blocks + 1):
                skip_h = skip_connections.pop() # Pop corresponding skip connection

                # Pad/Crop h to match skip_h's spatial dimension before concat
                if h.size(-1) != skip_h.size(-1):
                    target_len = skip_h.size(-1)
                    if h.size(-1) < target_len:
                        padding_diff = target_len - h.size(-1)
                        h = F.pad(h, (padding_diff // 2, padding_diff - padding_diff // 2))
                    else:
                        h = h[..., :target_len]

                h = torch.cat([h, skip_h], dim=1)

                # Now pass through the ResidualBlock
                current_res_block = self.up_blocks[up_block_idx]
                assert isinstance(current_res_block, ResidualBlock), "Block order mismatch in up_blocks"
                h = current_res_block(h, t_emb, cond_embedding)
                up_block_idx += 1

            # After all ResBlocks for this level, if not the highest resolution level, upsample.
            if i_level != 0: # If not the output resolution level
                if up_block_idx < len(self.up_blocks) and isinstance(self.up_blocks[up_block_idx], UpsampleBlock):
                    current_upsample_block = self.up_blocks[up_block_idx]
                    h = current_upsample_block(h)
                    up_block_idx += 1
                else:
                    # This implies an architectural mismatch if an UpsampleBlock was expected but not found,
                    # or if we run out of blocks.
                    if up_block_idx >= len(self.up_blocks):
                         logger.warning(f"Ran out of up_blocks at upsampling stage for level {i_level}. Current h shape: {h.shape}")
                    else:
                         logger.warning(f"Expected UpsampleBlock but found {type(self.up_blocks[up_block_idx])} at up_block_idx {up_block_idx} for level {i_level}.")


        h = self.final_norm(h)
        h = F.silu(h)
        out = self.final_conv(h)
        return out

if __name__ == '__main__':
    logger.info("Starting model diffusion.py example usage (with FiLM conditioning)...")

    batch_size = 2
    # Ensure audio_length is divisible by 2^(number of downsampling stages)
    # channel_mult=(1, 2, 2) -> 2 downsampling stages. Factor = 2*2 = 4
    audio_length = 2048 * 4
    in_audio_channels = 1
    out_audio_channels = 1
    base_model_channels = 32
    time_embedding_dim = 128 # Must be even

    cond_embedding_dim = 64 # Example dimension for the global conditioning vector

    unet_model = DiffusionUNet(
        in_channels=in_audio_channels, model_channels=base_model_channels,
        out_channels=out_audio_channels, time_emb_dim=time_embedding_dim,
        cond_emb_dim=cond_embedding_dim,
        channel_mult=(1, 2, 2), # 2 downsampling layers
        num_residual_blocks=1, # Simpler for test
        use_attention_at_resolution=(1,) # Attention at the middle resolution
    )
    logger.info(f"DiffusionUNet model (with FiLM) instantiated.")
    num_params = sum(p.numel() for p in unet_model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {num_params / 1e6:.2f} M")

    dummy_audio_input = torch.randn(batch_size, in_audio_channels, audio_length)
    dummy_timesteps = torch.randint(0, 100, (batch_size,)) # Example timesteps
    dummy_cond_embedding = torch.randn(batch_size, cond_embedding_dim)

    logger.info(f"Input audio shape: {dummy_audio_input.shape}")
    logger.info(f"Timesteps shape: {dummy_timesteps.shape}")
    logger.info(f"Cond_embedding shape: {dummy_cond_embedding.shape}")

    try:
        predicted_noise = unet_model(dummy_audio_input, dummy_timesteps, dummy_cond_embedding)
        logger.info(f"Predicted noise output shape: {predicted_noise.shape}")
        assert predicted_noise.shape == (batch_size, out_audio_channels, audio_length), \
            f"Output shape mismatch. Expected {(batch_size, out_audio_channels, audio_length)}, got {predicted_noise.shape}"
        logger.info("DiffusionUNet forward pass successful (with FiLM integration).")
    except ImportError:
        logger.error("Torch import failed. Cannot run U-Net forward pass. This is expected if torch is not installed.")
    except Exception as e:
        logger.error(f"Error during DiffusionUNet forward pass: {e}", exc_info=True)
        logger.error("This could be due to torch not being installed OR issues in U-Net skip connection/shape logic.")

    # Test unconditional model
    logger.info("Testing unconditional DiffusionUNet...")
    unet_unconditional = DiffusionUNet(cond_emb_dim=None, time_emb_dim=time_embedding_dim, model_channels=base_model_channels)
    try:
        predicted_noise_uncond = unet_unconditional(dummy_audio_input, dummy_timesteps, None) # Pass None for cond_embedding
        logger.info(f"Unconditional predicted noise output shape: {predicted_noise_uncond.shape}")
        assert predicted_noise_uncond.shape == (batch_size, out_audio_channels, audio_length)
        logger.info("Unconditional DiffusionUNet forward pass successful.")
    except ImportError:
        logger.error("Torch import failed for unconditional test.")
    except Exception as e:
        logger.error(f"Error during unconditional DiffusionUNet forward pass: {e}", exc_info=True)


    logger.info("model diffusion.py example usage finished.")
