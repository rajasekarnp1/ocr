import torch
import torch.nn as nn
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def get_noise_schedule(num_timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02, schedule_type: str = 'linear') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates a noise schedule for the diffusion process.

    Args:
        num_timesteps (int): The total number of diffusion steps.
        beta_start (float): The starting value for beta (variance of noise added at each step).
        beta_end (float): The ending value for beta.
        schedule_type (str): Type of schedule ('linear' or 'cosine').

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - betas: Noise variance at each timestep.
            - alphas: 1.0 - betas.
            - alphas_cumprod: Cumulative product of alphas.
            - alphas_cumprod_prev: Cumulative product of alphas up to the previous step.
    """
    if schedule_type == 'linear':
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
    elif schedule_type == 'cosine':
        s = 0.008
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps, dtype=torch.float32)
        alphas_cumprod_cosine = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod_cosine = alphas_cumprod_cosine / alphas_cumprod_cosine[0]
        betas = 1 - (alphas_cumprod_cosine[1:] / alphas_cumprod_cosine[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999)
    else:
        raise ValueError(f"Unsupported schedule_type: {schedule_type}. Choose 'linear' or 'cosine'.")

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

    logger.info(f"Generated '{schedule_type}' noise schedule with {num_timesteps} timesteps.")
    return betas, alphas, alphas_cumprod, alphas_cumprod_prev

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            logger.error("Dimension for TimeEmbedding must be an even number.")
            raise ValueError("Dimension must be an even number.")
        self.dim = dim
        logger.debug(f"TimeEmbedding initialized with dimension {dim}.")

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        # exponents = torch.arange(half_dim, dtype=torch.float32, device=device) / half_dim # Original
        # Corrected exponents to ensure it matches common implementations:
        exponents = torch.arange(half_dim, dtype=torch.float32, device=device)
        inv_freq = torch.exp(-math.log(10000.0) * (exponents / half_dim)) # Corrected application of log and division

        args = timesteps.float().unsqueeze(1) * inv_freq.unsqueeze(0)
        embedding = torch.cat((args.sin(), args.cos()), dim=-1)
        logger.debug(f"TimeEmbedding forward pass: input shape {timesteps.shape}, output shape {embedding.shape}")
        return embedding

class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        if dim % 2 != 0:
            logger.error("Dimension for PositionalEncoding must be an even number.")
            raise ValueError("Dimension must be an even number for sinusoidal positional encoding.")
        self.dim = dim
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Correct calculation for div_term, ensuring it matches standard implementations
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) # This will fail if dim is odd. Handled by check.

        pe = pe.unsqueeze(0) # Shape: (1, max_len, dim)
        self.register_buffer('pe', pe)
        logger.debug(f"PositionalEncoding initialized with dim {dim} and max_len {max_len}.")

    def forward(self, x_or_seq_len: [torch.Tensor, int]) -> torch.Tensor:
        """
        Args:
            x_or_seq_len (torch.Tensor or int):
                If Tensor, expected shape (batch, seq_len, features) or (seq_len, features) or (seq_len,).
                The positional encoding is returned for the seq_len dimension.
                If int, it's treated as seq_len.
        Returns:
            torch.Tensor: Positional encodings of shape (1, seq_len, dim).
        """
        seq_len: int
        if isinstance(x_or_seq_len, torch.Tensor):
            if x_or_seq_len.ndim == 1:       # (seq_len,)
                seq_len = x_or_seq_len.size(0)
            elif x_or_seq_len.ndim == 2:     # (batch_or_seq, seq_or_features) - assume (batch, seq_len) typically
                seq_len = x_or_seq_len.size(1)
            elif x_or_seq_len.ndim >= 3:     # (batch, seq_len, features)
                seq_len = x_or_seq_len.size(1)
            else: # Should not happen for ndim > 0
                raise ValueError("Unsupported tensor shape for inferring seq_len.")
        elif isinstance(x_or_seq_len, int):
            seq_len = x_or_seq_len
        else:
            logger.error(f"Unsupported input type for PositionalEncoding: {type(x_or_seq_len)}")
            raise TypeError("Input must be Tensor or int to determine sequence length.")

        if seq_len > self.pe.size(1):
             logger.warning(f"Requested seq_len {seq_len} > max_len {self.pe.size(1)}. Returning max_len encoding and attempting to slice.")
             # This case should ideally not happen if max_len is set appropriately.
             # If it does, we return the maximum available length.
             return self.pe[:, :self.pe.size(1), :]

        logger.debug(f"PositionalEncoding forward: requested seq_len {seq_len}. Output shape (1, {seq_len}, {self.dim})")
        return self.pe[:, :seq_len, :]

if __name__ == '__main__':
    logger.info("Starting model utils.py example usage...")
    num_steps = 100
    logger.info(f"Testing Noise Schedule (Linear, {num_steps} steps)")
    betas_lin, alphas_lin, alphas_cumprod_lin, alphas_cumprod_prev_lin = get_noise_schedule(num_steps, schedule_type='linear')
    assert alphas_cumprod_prev_lin[0] == 1.0, "First value of alphas_cumprod_prev should be 1.0"
    assert betas_lin.shape == (num_steps,), f"Betas shape mismatch: {betas_lin.shape}"
    logger.info(f"Linear schedule shapes: betas {betas_lin.shape}, alphas_cumprod {alphas_cumprod_lin.shape}")

    logger.info(f"Testing Noise Schedule (Cosine, {num_steps} steps)")
    betas_cos, alphas_cos, alphas_cumprod_cos, alphas_cumprod_prev_cos = get_noise_schedule(num_steps, schedule_type='cosine')
    assert alphas_cumprod_prev_cos[0] == 1.0, "First value of alphas_cumprod_prev_cos for cosine schedule should be 1.0"
    assert betas_cos.shape == (num_steps,), f"Cosine betas shape mismatch: {betas_cos.shape}"
    logger.info(f"Cosine schedule shapes: betas {betas_cos.shape}, alphas_cumprod {alphas_cumprod_cos.shape}")

    time_dim = 32 # Must be even
    time_embed_module = TimeEmbedding(dim=time_dim)
    example_timesteps = torch.randint(0, num_steps, (4,)) # Batch of 4 timesteps
    time_embeddings = time_embed_module(example_timesteps)
    assert time_embeddings.shape == (4, time_dim), f"TimeEmbedding output shape mismatch: {time_embeddings.shape}"
    logger.info(f"TimeEmbedding test passed. Output shape: {time_embeddings.shape}")

    pos_dim = 32 # Must be even for this PositionalEncoding implementation
    max_seq_len_test = 50
    pos_encode_module = PositionalEncoding(dim=pos_dim, max_len=max_seq_len_test)

    test_seq_len_tensor = 20
    dummy_input_for_pos_tensor = torch.randn(4, test_seq_len_tensor, pos_dim) # (batch, seq_len, features)
    positional_encodings_from_tensor = pos_encode_module(dummy_input_for_pos_tensor)
    assert positional_encodings_from_tensor.shape == (1, test_seq_len_tensor, pos_dim), f"PositionalEncoding from tensor input shape mismatch: {positional_encodings_from_tensor.shape}"

    test_seq_len_int = 25
    positional_encodings_from_int = pos_encode_module(test_seq_len_int)
    assert positional_encodings_from_int.shape == (1, test_seq_len_int, pos_dim), f"PositionalEncoding from int input shape mismatch: {positional_encodings_from_int.shape}"

    logger.info(f"PositionalEncoding test passed. Output shapes: from tensor {positional_encodings_from_tensor.shape}, from int {positional_encodings_from_int.shape}")

    # Test PositionalEncoding with seq_len > max_len
    too_long_seq_len = max_seq_len_test + 10
    logger.info(f"Testing PositionalEncoding with seq_len ({too_long_seq_len}) > max_len ({max_seq_len_test})...")
    pos_enc_too_long = pos_encode_module(too_long_seq_len)
    assert pos_enc_too_long.shape == (1, max_seq_len_test, pos_dim), f"PositionalEncoding for too long seq_len shape mismatch: {pos_enc_too_long.shape}"
    logger.info(f"PositionalEncoding for too long sequence handled. Output shape: {pos_enc_too_long.shape}")

    logger.info("model utils.py example usage finished.")
