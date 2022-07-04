import copy
import torch
from torch import nn


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def with_pos_embed(tensor, pos):
    return tensor if pos is None else tensor + pos


def get_reference_points(spatial_shapes, batch_size, device):
    reference_points_list = []  # [(2, 7832, 2), (2, 1980, 2), (2, 506, 2), (2, 132, 2)]
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        # (H_, W_), (H_, W_)
        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                      torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1).expand((batch_size, H_ * W_)) / H_  # (2, H_ * W_)
        ref_x = ref_x.reshape(-1).expand((batch_size, H_ * W_)) / W_  # (2, H_ * W_)
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)  # (2, H_ * W_, 2)
    reference_points = torch.cat(reference_points_list, 1)  # (2, 10540, 2)
    # reference_points[:, :, None] (2, 10450, 1, 2)  valid_ratios[:, None] (2, 1, 4, 2)
    _, xx, yy = reference_points.shape
    reference_points = reference_points[:, :, None].expand((batch_size, xx, lvl + 1, yy))  # (2, 10540, 4, 2)
    return reference_points
