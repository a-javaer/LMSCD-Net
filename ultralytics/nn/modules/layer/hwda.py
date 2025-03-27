import torch
import torch.nn as nn
import torch.nn.functional as F



def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
    

class mn_conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__()
        padding = 0 if k == s else autopad(k, p, d)
        self.c = nn.Conv2d(c1, c2, k, s, padding, groups=g)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.c(x)))


class InvertedBottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, stride=1, e=None):
        # input_channels, output_channels, repetition, stride, expension ratio
        super().__init__()
        # act = nn.ReLU6(inplace=True) if NL=="RE" else nn.Hardswish()
        c_mid = e if e != None else c1

        features = [mn_conv(c1, c_mid)]
        features.extend(
            [
                mn_conv(c_mid, c_mid, k, stride, g=c_mid),
                nn.Conv2d(c_mid, c2, 1),
                nn.BatchNorm2d(c2),
                # nn.SiLU(),
            ]
        )
        self.layers = nn.Sequential(*features)

    def forward(self, x):
            return self.layers(x)



class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class GlobalExtraction(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = 2 * n_feats
        d_feats = n_feats // 2
        self.n_feats = n_feats
        self.d_feats = d_feats
        self.i_feats = i_feats

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        # Multiscale Large Kernel Attention
        self.LKA7 = nn.Sequential(
            nn.Conv2d(d_feats, d_feats, 7, 1, 7 // 2, groups=d_feats),
            nn.Conv2d(d_feats, d_feats, 9, stride=1, padding=(9 // 2) * 4, groups=d_feats, dilation=4),
            nn.Conv2d(d_feats, d_feats, 1, 1, 0))
        self.LKA5 = nn.Sequential(
            nn.Conv2d(d_feats, d_feats, 5, 1, 5 // 2, groups=d_feats),
            nn.Conv2d(d_feats, d_feats, 7, stride=1, padding=(7 // 2) * 3, groups=d_feats, dilation=3),
            nn.Conv2d(d_feats, d_feats, 1, 1, 0))
        self.LKA3 = nn.Sequential(
            nn.Conv2d(d_feats, d_feats, 3, 1, 1, groups=d_feats),
            nn.Conv2d(d_feats, d_feats, 5, stride=1, padding=(5 // 2) * 2, groups=d_feats, dilation=2),
            nn.Conv2d(d_feats, d_feats, 1, 1, 0))

        self.X3 = nn.Conv2d(d_feats, d_feats, 3, 1, 1, groups=d_feats)
        self.X5 = nn.Conv2d(d_feats, d_feats, 5, 1, 5 // 2, groups=d_feats)
        self.X7 = nn.Conv2d(d_feats, d_feats, 7, 1, 7 // 2, groups=d_feats)

        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(i_feats, n_feats, 1, 1, 0))

    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.proj_first(x)
        x = list(torch.split(x, self.d_feats, dim=1))
        x[1] = self.LKA3(x[1]) * self.X3(x[1])
        x[2] = self.LKA5(x[2]) * self.X5(x[2])
        x[3] = self.LKA7(x[3]) * self.X7(x[3])
        x = torch.cat(x, dim=1)
        x = self.proj_last(x) * self.scale + shortcut
        return x


class ContextExtraction(nn.Module):
    """ Inception depthweise convolution
    """

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.25):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.p = nn.AvgPool2d(3, stride=1, padding=1)
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)


    def forward(self, x):
        x = self.p(x)
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        x = torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )
        
        return x


class MultiscaleFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.local= ContextExtraction(dim)
        self.global_ = GlobalExtraction(dim)
        self.bn = nn.BatchNorm2d(num_features=dim)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0))

    def forward(self, x):
        x1 = self.local(x)
        x2 = self.global_(x)

        fuse = self.bn(x1 + x2)
        fuse = self.proj(fuse)
        return fuse

 # MAB
class HWDA(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=2):
        super().__init__()

        self.m1 = InvertedBottleneck(inp, oup, k=kernel_size, stride=stride, e=2 * oup)
        self.LKA = MultiscaleFusion(oup)

    def forward(self, x):
        x = self.m1(x)
        # large kernel attention
        x = self.LKA(x)
        return x