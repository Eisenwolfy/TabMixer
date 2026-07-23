import torch
import torch.nn as nn

"""
B - batch size
F - number of features
D - embedding size
"""

class NumericalEmbedding(nn.Module):
    """Periodic (sin/cos) embedding + positional embedding per feature."""
    def __init__(self, F, D):
        super().__init__()
        assert D % 2 == 0
        # Reverted back to a safe random normal init scale so angles don't immediately collapse to 0
        self.freq = nn.Parameter(torch.randn(F, D // 2))
        self.phase = nn.Parameter(torch.zeros(F, D // 2))
        self.pos_embed = nn.Parameter(torch.randn(F, D) * 0.02)

    def forward(self, x):
        # x: (B, F)
        angle = x.unsqueeze(-1) * self.freq + self.phase # (B, F, D/2)
        emb = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)  # (B, F, D)
        return emb + self.pos_embed


class TokenMixer(nn.Module):
    def __init__(self, F, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(F, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, F),
        )

    def forward(self, x):
        x_t = x.transpose(1, 2)
        mixed = self.net(x_t)
        return mixed.transpose(1, 2)


class ChannelMixer(nn.Module):
    def __init__(self, D, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, D),
        )

    def forward(self, x):
        return self.net(x)


class FeatureGate(nn.Module):
    def __init__(self, F, D):
        super().__init__()
        self.local_proj = nn.Linear(D, D)
        # Mixes across features (F) for each channel efficiently
        self.context_proj = nn.Linear(F, F)
        self.to_gate = nn.Linear(D, D)
        self.last_gate = None

    def forward(self, x):
        # x shape: (B, F, D)
        local = self.local_proj(x)
        context = self.context_proj(x.transpose(1, 2)).transpose(1, 2)

        gate = torch.sigmoid(self.to_gate(local + context))  # (B, F, D)
        self.last_gate = gate.detach()
        return gate


class MixerBlock(nn.Module):
    def __init__(self, F, D, hidden_dim, drop_prob=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(D)
        self.token_mixer = TokenMixer(F, hidden_dim)
        self.gate1 = FeatureGate(F, D)

        self.norm2 = nn.LayerNorm(D)
        self.channel_mixer = ChannelMixer(D, hidden_dim)
        self.gate2 = FeatureGate(F, D)

        self.drop_prob = drop_prob

    def _highway(self, x_old, x_new, gate):
        # Calculate the raw gated update
        update = gate * (x_new - x_old)

        if self.training and self.drop_prob > 0:
            B = x_old.shape[0]
            keep_prob = 1 - self.drop_prob
            # Draw mask per sample
            keep_mask = (torch.rand(B, 1, 1, device=x_old.device) < keep_prob).float()

            # Scale the actual residual update vector, leaving gate boundaries intact
            update = update * (keep_mask / keep_prob)

        return x_old + update

    def forward(self, x):
        mixed = self.token_mixer(self.norm1(x))
        gate1 = self.gate1(mixed)
        x = self._highway(x, mixed, gate1)

        channeled = self.channel_mixer(self.norm2(x))
        gate2 = self.gate2(channeled)
        x = self._highway(x, channeled, gate2)
        return x


class TabMixer(nn.Module):
    def __init__(self, F, D, hidden_dim=None, n_blocks=3, n_classes=2, pooling="mean"):
        super().__init__()
        hidden_dim = hidden_dim or 2 * F
        self.embed = NumericalEmbedding(F, D)
        self.blocks = nn.ModuleList([
            MixerBlock(F, D, hidden_dim) for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(D)
        self.pooling = pooling
        head_in = D if pooling == "mean" else F * D
        self.head = nn.Linear(head_in, n_classes)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1) if self.pooling == "mean" else x.flatten(1)
        return self.head(x)

    def get_gate_values(self):
        out = []
        for i, block in enumerate(self.blocks):
            out.append({
                "block": i,
                "gate1": block.gate1.last_gate,
                "gate2": block.gate2.last_gate,
                # Cleaned up old attention tracking keys
            })
        return out


if __name__ == "__main__":
    B, F, D, n_blocks, n_classes = 8, 20, 32, 3, 5
    model = TabMixer(F, D, n_blocks=n_blocks, n_classes=n_classes)
    x = torch.randn(B, F)

    model.train()
    print("train shape:", model(x).shape)

    model.eval()
    out = model(x)
    print("eval shape:", out.shape)

    gates = model.get_gate_values()
    print("gate1 block0 shape:", gates[0]["gate1"].shape)