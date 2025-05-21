import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteDNN(nn.Module):
    """
    A lightweight feed-forward MLP for discrete diffusion reconstruction.
    Designed to predict masked (zeroed) inputs given partially corrupted vectors.
    """

    def __init__(self, input_dim, hidden_dim=128, output_dim=None, use_time=False, time_dim=64, dropout=0.2):
        super(DiscreteDNN, self).__init__()
        self.use_time = use_time
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.time_dim = time_dim

        total_input_dim = input_dim + (time_dim if use_time else 0)

        self.fc1 = nn.Linear(total_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.output_dim)

        self.drop = nn.Dropout(dropout)

        if self.use_time:
            self.time_embed = nn.Linear(time_dim, time_dim)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t=None):
        """
        x: [batch_size, input_dim]
        t: [batch_size] (optional timestep indices if use_time=True)
        """
        if self.use_time:
            assert t is not None, "Timesteps required when use_time=True"
            t_emb = timestep_embedding(t, self.time_dim).to(x.device)
            t_emb = self.time_embed(t_emb)
            x = torch.cat([x, t_emb], dim=-1)

        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Sinusoidal timestep embedding (same as BERT-style positional embedding).

    Args:
        timesteps: Tensor of shape [B]
        dim: embedding dimension
    Returns:
        Tensor of shape [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(0, half, dtype=torch.float32) * (torch.log(torch.tensor(max_period)) / half)
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
