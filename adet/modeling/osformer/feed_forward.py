import torch
from torch import nn


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, padding=1)
        )

    def forward(self, src, spatial_shapes, *args):
        split_list = [(w * h) for (w, h) in spatial_shapes]
        feat_levels = []
        for memory, (w, h) in zip(src.split(split_list, 1), spatial_shapes):
            memory = memory.view(-1, w, h, self.d_model).permute(0, 3, 1, 2)
            memory = self.ffn(memory)
            feat_levels.append(memory.flatten(2).transpose(1, 2))
        return torch.cat(feat_levels, 1)


class VanillaFeedForwardNetwork(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1, bias=False),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 3, padding=1, bias=False)
        )

    def forward(self, src, *args):
        return self.ffn(src.permute(0, 2, 1)).permute(0, 2, 1)


class StdFeedForwardNetwork(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, *args):
        return self.norm(src + self.ffn(src))


def get_ffn(d_model, ffn_type):
    if ffn_type == 'std':
        return StdFeedForwardNetwork(d_model)
    if ffn_type == 'vanilla':
        return VanillaFeedForwardNetwork(d_model)
    return FeedForwardNetwork(d_model)
