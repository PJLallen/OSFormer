from torch import nn


class DCIN(nn.Module):
    def __init__(self, num_kernels, norm):
        super(DCIN, self).__init__()

        self.affine_scale = nn.Linear(num_kernels, num_kernels, bias=True)
        self.affine_bias = nn.Linear(num_kernels, 1, bias=True)
        self.norm = norm

    def forward(self, mask_features, kernel_features):
        """
        mask_features: shape of (1, c, w, h)
        kernel_features: shape of (n, c)

        return: shape of (1, n, w, h)
        """
        kernel_w = self.affine_scale(kernel_features)  # (n, c)
        kernel_b = self.affine_bias(kernel_features)  # (n, 1)
        bs, c, w, h = mask_features.shape
        x = mask_features.view((bs, c, -1))  # (bs, c, k)
        if self.norm:
            x_mean = x.mean(2, keepdim=True)  # (bs, c, 1)
            x_centered = x - x_mean  # (bs, c, k)
            # add 1e-10 to avoid NaN
            x_std_rev = ((x_centered * x_centered).mean(2, keepdim=True) + 1e-10).rsqrt()  # (bs, c, 1)
            x_norm = x_centered * x_std_rev  # (bs, c, k)
        else:
            x_norm = x

        return (kernel_w.matmul(x_norm) + kernel_b).view((bs, -1, w, h))
