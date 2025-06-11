# src/model/condition.py
"""
Modules for processing conditioning information (e.g., text embeddings, mel spectrograms for style).
"""
import torch
import torch.nn as nn
# import librosa # If processing audio for conditioning, e.g. mel spectrogram

class TextConditionEncoder(nn.Module):
    def __init__(self, text_embedding_dim, output_dim, num_layers=2, hidden_dim=512):
        """
        Encodes text embeddings (e.g., from a pre-trained text model like BERT, CLIP)
        into a conditioning vector for the diffusion model.

        Args:
            text_embedding_dim (int): Dimension of the input text embeddings.
            output_dim (int): Dimension of the output conditioning vector.
            num_layers (int): Number of linear layers.
            hidden_dim (int): Hidden dimension for intermediate layers.
        """
        super().__init__()
        self.text_embedding_dim = text_embedding_dim
        self.output_dim = output_dim

        layers = []
        current_dim = text_embedding_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU()) # or nn.GELU()
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))

        self.encoder = nn.Sequential(*layers)
        print(f"TextConditionEncoder initialized: {text_embedding_dim} -> {output_dim}")

    def forward(self, text_embedding):
        """
        Args:
            text_embedding (torch.Tensor): Input text embeddings (batch_size, text_embedding_dim)
                                           or (batch_size, seq_len, text_embedding_dim).
                                           If sequence, it might need pooling first.
        """
        if text_embedding.dim() > 2:
            # Example: Average pooling if input is (batch, seq_len, dim)
            text_embedding = text_embedding.mean(dim=1)

        condition_vector = self.encoder(text_embedding)
        return condition_vector

class MelSpectrogramConditionEncoder(nn.Module):
    def __init__(self, n_mels, output_dim, num_conv_layers=3, conv_channels=None,
                 conv_kernel_size=3, use_rnn=False, rnn_hidden_dim=256):
        """
        Encodes a mel spectrogram (e.g., for style transfer or voice cloning)
        into a conditioning vector.

        Args:
            n_mels (int): Number of mel bands in the input spectrogram.
            output_dim (int): Dimension of the output conditioning vector.
            num_conv_layers (int): Number of 1D convolutional layers.
            conv_channels (list/tuple, optional): Output channels for each conv layer.
                                                 If None, defaults to [128, 256, output_dim].
            conv_kernel_size (int): Kernel size for convolutions.
            use_rnn (bool): Whether to use an RNN (e.g., GRU) after convolutions for global features.
            rnn_hidden_dim (int): Hidden dimension for the RNN.
        """
        super().__init__()
        self.n_mels = n_mels
        self.output_dim = output_dim
        self.use_rnn = use_rnn

        if conv_channels is None:
            # Default channel sizes, ensure last one matches output_dim if no RNN
            _ch = [64, 128, output_dim if not use_rnn else rnn_hidden_dim]
            conv_channels = _ch[:num_conv_layers]
            if len(conv_channels) < num_conv_layers : # if num_conv_layers is small
                 conv_channels += [output_dim if not use_rnn else rnn_hidden_dim] * (num_conv_layers - len(conv_channels))


        conv_layers = []
        current_channels = n_mels
        for i in range(num_conv_layers):
            conv_layers.append(
                nn.Conv1d(current_channels, conv_channels[i],
                          kernel_size=conv_kernel_size, padding=conv_kernel_size//2)
            )
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.BatchNorm1d(conv_channels[i])) # Optional
            current_channels = conv_channels[i]

        self.conv_encoder = nn.Sequential(*conv_layers)

        if self.use_rnn:
            self.rnn = nn.GRU(current_channels, rnn_hidden_dim, batch_first=True)
            self.fc = nn.Linear(rnn_hidden_dim, output_dim)
        elif current_channels != output_dim:
            # If no RNN, and conv output dim doesn't match target, add a final linear layer
            self.final_fc = nn.Linear(current_channels, output_dim)
        else:
            self.final_fc = nn.Identity()


        print(f"MelSpectrogramConditionEncoder initialized: n_mels={n_mels} -> output_dim={output_dim}")
        print(f"  Conv layers: {num_conv_layers}, Channels: {conv_channels}, Kernel: {conv_kernel_size}")
        if use_rnn: print(f"  Using RNN with hidden_dim: {rnn_hidden_dim}")


    def forward(self, mel_spectrogram):
        """
        Args:
            mel_spectrogram (torch.Tensor): Input mel spectrogram (batch_size, n_mels, time_frames).
        """
        # mel_spectrogram is (batch, n_mels, time_frames)
        # Conv1D expects (batch, channels, length), so n_mels is channels.
        x = self.conv_encoder(mel_spectrogram) # (batch, conv_out_channels, time_frames)

        if self.use_rnn:
            # For RNN, typically we want (batch, seq, features)
            x = x.transpose(1, 2) # (batch, time_frames, conv_out_channels)
            _, h_n = self.rnn(x)  # h_n is (num_layers*num_directions, batch, rnn_hidden_dim)
            x = h_n.squeeze(0)    # (batch, rnn_hidden_dim) - taking last hidden state
            condition_vector = self.fc(x)
        else:
            # Global average pooling over time dimension
            x = torch.mean(x, dim=2) # (batch, conv_out_channels)
            condition_vector = self.final_fc(x)

        return condition_vector

# Example of a generic conditioning embedder that can be used for discrete classes
class ClassConditionEncoder(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        """
        Simple embedding layer for class-conditional generation.
        Args:
            num_classes (int): Total number of classes.
            embedding_dim (int): Desired dimension for the class embedding.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        print(f"ClassConditionEncoder initialized: num_classes={num_classes} -> embedding_dim={embedding_dim}")

    def forward(self, class_labels):
        """
        Args:
            class_labels (torch.Tensor): Tensor of class indices (batch_size,).
        Returns:
            torch.Tensor: Class embeddings (batch_size, embedding_dim).
        """
        return self.embedding(class_labels)


if __name__ == '__main__':
    # Example Usage

    # 1. Text Condition Encoder
    print("\n--- Text Condition Encoder Example ---")
    dummy_text_embed_dim = 768 # e.g., from BERT base
    text_cond_output_dim = 256
    text_encoder = TextConditionEncoder(dummy_text_embed_dim, text_cond_output_dim)

    batch_size = 4
    # Example with (batch, dim)
    dummy_text_embeddings_simple = torch.randn(batch_size, dummy_text_embed_dim)
    text_condition = text_encoder(dummy_text_embeddings_simple)
    print(f"Text condition (simple) output shape: {text_condition.shape}") # Expected: (batch_size, text_cond_output_dim)

    # Example with (batch, seq_len, dim)
    seq_len = 10
    dummy_text_embeddings_seq = torch.randn(batch_size, seq_len, dummy_text_embed_dim)
    text_condition_seq = text_encoder(dummy_text_embeddings_seq)
    print(f"Text condition (sequence) output shape: {text_condition_seq.shape}")


    # 2. Mel Spectrogram Condition Encoder
    print("\n--- Mel Spectrogram Condition Encoder Example ---")
    n_mels_param = 80
    mel_cond_output_dim = 256
    time_frames = 100 # Example number of time frames in the spectrogram

    # With RNN
    mel_encoder_rnn = MelSpectrogramConditionEncoder(n_mels_param, mel_cond_output_dim, use_rnn=True, num_conv_layers=3)
    dummy_mel_spec = torch.randn(batch_size, n_mels_param, time_frames)
    mel_condition_rnn = mel_encoder_rnn(dummy_mel_spec)
    print(f"Mel condition (RNN) output shape: {mel_condition_rnn.shape}") # Expected: (batch_size, mel_cond_output_dim)

    # Without RNN (using global average pooling after convs)
    mel_encoder_pool = MelSpectrogramConditionEncoder(n_mels_param, mel_cond_output_dim, use_rnn=False, num_conv_layers=4)
    mel_condition_pool = mel_encoder_pool(dummy_mel_spec)
    print(f"Mel condition (Pooling) output shape: {mel_condition_pool.shape}")


    # 3. Class Condition Encoder
    print("\n--- Class Condition Encoder Example ---")
    num_speaker_classes = 10
    speaker_embedding_dim = 128
    class_encoder = ClassConditionEncoder(num_speaker_classes, speaker_embedding_dim)

    dummy_speaker_ids = torch.randint(0, num_speaker_classes, (batch_size,))
    speaker_condition = class_encoder(dummy_speaker_ids)
    print(f"Speaker ID condition output shape: {speaker_condition.shape}") # Expected: (batch_size, speaker_embedding_dim)
