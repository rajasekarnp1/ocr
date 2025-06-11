# src/model/utils.py
"""
Utility functions and helper classes for the model components.
e.g., Positional Encodings, noise schedules, activation functions.
"""
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    From PyTorch Transformer tutorial:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe) # Not a model parameter

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model)
                              or (batch_size, d_model) if used for single timestep.
        """
        if x.dim() == 3: # (seq_len, batch_size, d_model)
            x = x + self.pe[:x.size(0), :]
        elif x.dim() == 2: # (batch_size, d_model) - for timestep embedding
            # Assuming x is (batch_size, d_model) and we want to add PE for each item in batch
            # This might need adjustment based on how timesteps are passed.
            # If 'x' here represents timesteps (e.g., tensor of [t1, t2...]) then this is different.
            # This current PE is more for sequence positions.
            # For single timesteps, a simpler embedding or a different PE might be used.
            # For now, let's assume we are adding to an existing embedding.
            # This part might need to be context-specific for diffusion model timesteps.
            # If x is (batch_size, features) and we want to add PE based on a position for each:
            # This application is a bit ambiguous for typical diffusion time embedding.
            # Usually, time 't' is directly embedded.
            # Let's assume this PE is for sequence elements, not direct time embedding.
            # Consider a dedicated TimeEmbedding class for diffusion timesteps.
            pass # This PE is more for sequence models.
        return self.dropout(x)


class TimeEmbedding(nn.Module):
    """
    Embeds a scalar timestep into a vector of a specified dimension.
    Commonly used in diffusion models.
    """
    def __init__(self, time_dim, embedding_dim):
        super().__init__()
        # time_dim is usually the max number of timesteps, but here it's used as input feature dim for MLP
        # For sinusoidal, it's d_model (embedding_dim)
        # Let's make this a learned MLP following common practice for DDPM.
        self.embedding_dim = embedding_dim
        half_dim = embedding_dim // 2
        # Sinusoidal part
        self.freqs = nn.Parameter(torch.exp(torch.arange(0, half_dim, dtype=torch.float32) * -(math.log(10000.0) / (half_dim -1))))

        # Optional MLP part (can be added for more expressiveness)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4), # Or use half_dim * 2 if only sinusoidal part is fed
            nn.Mish(), # Or SiLU, GELU
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        print(f"TimeEmbedding initialized: input scalar t -> output_dim {embedding_dim}")

    def forward(self, t):
        """
        Args:
            t (torch.Tensor): Scalar tensor of timesteps (batch_size,).
        Returns:
            torch.Tensor: Time embeddings (batch_size, embedding_dim).
        """
        t_unsqueezed = t.float().unsqueeze(-1) # (batch_size, 1)
        args = t_unsqueezed * self.freqs.unsqueeze(0) # (batch_size, half_dim)
        sinusoidal_embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1) # (batch_size, embedding_dim)

        # Pass through MLP
        time_embedding = self.mlp(sinusoidal_embedding)
        return time_embedding


def get_noise_schedule(schedule_type, num_timesteps, beta_start=1e-4, beta_end=0.02, device=None):
    """
    Generates parameters for the noise schedule (betas, alphas, etc.).

    Args:
        schedule_type (str): 'linear', 'cosine', 'quadratic', 'sigmoid'.
        num_timesteps (int): Total number of diffusion steps.
        beta_start (float): Starting value of beta.
        beta_end (float): Ending value of beta.
        device (torch.device, optional): Device to put tensors on.

    Returns:
        A tuple of tensors:
        betas, alphas, alphas_cumprod, alphas_cumprod_prev,
        sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
        log_one_minus_alphas_cumprod, sqrt_recip_alphas_cumprod,
        sqrt_recipm1_alphas_cumprod, posterior_variance, posterior_log_variance_clipped,
        posterior_mean_coef1, posterior_mean_coef2
    """
    if schedule_type == 'linear':
        betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device, dtype=torch.float64)
    elif schedule_type == 'cosine':
        # From "Improved Denoising Diffusion Probabilistic Models" https://arxiv.org/abs/2102.09672
        s = 0.008
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps, device=device, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999) # Paper uses 0.0001 and 0.999
    elif schedule_type == 'quadratic':
        betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps, device=device, dtype=torch.float64) ** 2
    elif schedule_type == 'sigmoid':
        betas_sig = torch.linspace(-6, 6, num_timesteps, device=device, dtype=torch.float64)
        betas = torch.sigmoid(betas_sig) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(f"Unknown noise schedule: {schedule_type}")

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device, dtype=torch.float64), alphas_cumprod[:-1]])

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    log_one_minus_alphas_cumprod = torch.log(1.0 - alphas_cumprod)
    sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)

    # Calculations for posterior q(x_{t-1} | x_t, x_0)
    # (Used in DDPM sampling)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_log_variance_clipped = torch.log(
        torch.cat([posterior_variance[1:2], posterior_variance[1:]]) # Ensure not log(0) for first step
    ) if len(posterior_variance) > 1 else torch.tensor([]) # Handle num_timesteps=1 case

    posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

    return (betas.float(), alphas.float(), alphas_cumprod.float(), alphas_cumprod_prev.float(),
            sqrt_alphas_cumprod.float(), sqrt_one_minus_alphas_cumprod.float(),
            log_one_minus_alphas_cumprod.float(), sqrt_recip_alphas_cumprod.float(),
            sqrt_recipm1_alphas_cumprod.float(), posterior_variance.float(),
            posterior_log_variance_clipped.float(), posterior_mean_coef1.float(),
            posterior_mean_coef2.float())


if __name__ == '__main__':
    # Example Usage

    # 1. Positional Encoding (more for sequence models, but included for completeness)
    print("--- Positional Encoding Example ---")
    d_model_pe = 64
    pe = PositionalEncoding(d_model_pe)
    seq_len = 10
    batch_size_pe = 4
    dummy_seq_data = torch.randn(seq_len, batch_size_pe, d_model_pe)
    encoded_seq = pe(dummy_seq_data)
    print(f"Positional encoded sequence shape: {encoded_seq.shape}")

    # 2. Time Embedding for diffusion models
    print("\n--- Time Embedding Example ---")
    time_emb_dim = 128
    time_embedder = TimeEmbedding(time_dim=None, embedding_dim=time_emb_dim) # time_dim not directly used in this MLP version's init
    batch_size_te = 8
    dummy_timesteps = torch.randint(0, 1000, (batch_size_te,)) # Example timesteps
    time_embeddings = time_embedder(dummy_timesteps)
    print(f"Time embeddings shape: {time_embeddings.shape}") # Expected: (batch_size_te, time_emb_dim)

    # 3. Noise Schedule
    print("\n--- Noise Schedule Example ---")
    num_diffusion_timesteps = 1000

    print("\nLinear Schedule:")
    schedule_params_linear = get_noise_schedule('linear', num_diffusion_timesteps)
    # (betas, alphas, alphas_cumprod, alphas_cumprod_prev, ...)
    print(f"Betas shape: {schedule_params_linear[0].shape}")
    print(f"Alphas_cumprod shape: {schedule_params_linear[2].shape}")
    print(f"First beta: {schedule_params_linear[0][0]}, Last beta: {schedule_params_linear[0][-1]}")

    print("\nCosine Schedule:")
    schedule_params_cosine = get_noise_schedule('cosine', num_diffusion_timesteps)
    print(f"Betas shape: {schedule_params_cosine[0].shape}")
    print(f"Alphas_cumprod shape: {schedule_params_cosine[2].shape}")
    print(f"First beta: {schedule_params_cosine[0][0]}, Last beta: {schedule_params_cosine[0][-1]}")

    # Check if all returned tensors have the correct length
    all_params_len_ok = all(p.shape[0] == num_diffusion_timesteps for i, p in enumerate(schedule_params_cosine) if i not in [10] or p.numel() > 0) # posterior_log_variance_clipped can be shorter or empty
    if schedule_params_cosine[10].numel() == 0 and num_diffusion_timesteps == 1: # special case for T=1
        pass
    elif schedule_params_cosine[10].shape[0] != num_diffusion_timesteps and num_diffusion_timesteps > 1 :
         all_params_len_ok = False # if not empty and not T=1, it should match T

    print(f"All schedule parameters have correct length for {num_diffusion_timesteps} timesteps: {all_params_len_ok}")

    # Test with num_timesteps = 1 for posterior_log_variance_clipped
    print("\nTest with T=1 (for posterior_log_variance_clipped edge case):")
    schedule_params_one_step = get_noise_schedule('linear', 1)
    print(f"Posterior log variance for T=1: {schedule_params_one_step[10]}, numel: {schedule_params_one_step[10].numel()}")
