import torch
import torch.nn as nn
import skimage
import numpy as np
from skimage import filters

from torch.autograd import Function


def thershold_mask(X, mean=[0.5], std=[0.5]):
    device = X.device
    std = torch.Tensor(std).to(device)[:, None]
    mean = torch.Tensor(mean).to(device)[:, None]
    X = X.clone().mul_(std).add_(mean).clamp(0, 1)
    X_fil = np.copy(X.clone().cpu().numpy())

    blurred_image = filters.gaussian(X_fil, sigma=1.0)
    t = filters.threshold_otsu(blurred_image)
    soft_mask = np.where(blurred_image > t, 1, blurred_image)
    soft_mask = torch.Tensor(soft_mask).to(device)
    return soft_mask


class MaskOperator(Function):

    @staticmethod
    def forward(ctx, x, use_thermal=False):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.use_thermal = use_thermal
        ctx.set_materialize_grads(False)
        
        x = x.clone()
        
        device = x.device
        
        x_rgb = []
        for x_i in x:
            mask = thershold_mask(x_i[0])

            x_rgba = torch.zeros_like(x_i)
            x_rgba[0:3] = x_i[-3::]
            x_rgba[3] = mask
            x_rgba = x_rgba.permute(1, 2, 0).cpu().numpy()
            x_rgb.append(torch.Tensor(skimage.color.rgba2rgb(x_rgba)).permute(2, 0, 1).to(device))
        
        x_rgb = torch.stack(x_rgb)

        if use_thermal:
            x[:, -3::] = x_rgb
            return x
        else:
            return x_rgb

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_output, None


mask_opt = MaskOperator.apply


class MaskLayer(nn.Module):
    def __init__(self, use_thermal=False):
        super(MaskLayer, self).__init__()
        self.use_thermal = use_thermal

    def forward(self, x):
        return mask_opt(x, self.use_thermal)