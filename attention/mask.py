import torch
import torch.nn as nn
import skimage
import numpy as np

class MaskLayer(nn.Module):
    def __init__(self, use_thermal=False):
        super(MaskLayer, self).__init__()
        self.use_thermal = use_thermal

    @staticmethod
    def thershold_mask(X, mean=[0.5], std=[0.5]):
        device = X.device
        std = torch.tensor(std).to(device)[:, None]
        mean = torch.tensor(mean).to(device)[:, None]
        X = X.clone().mul_(std).add_(mean).clamp(0, 1)
        #     X = X.permute(1, 2, 0)
        X_fil = np.copy(X.clone().cpu().numpy())

        blurred_image = skimage.filters.gaussian(X_fil, sigma=1.0)
        t = skimage.filters.threshold_otsu(blurred_image)
        #         binary_mask = blurred_image > t
        soft_mask = np.where(blurred_image > t, 1, blurred_image)
        soft_mask = torch.tensor(soft_mask).to(device)
        return soft_mask

    def forward(self, x):
        x = x.clone()
        
        device = x.device
        
        x_rgb = []
        
        for x_i in x:
        
            mask = self.thershold_mask(x_i[0])

            x_rgba = torch.zeros_like(x_i)
            # x_rgba[0:3] = inputs[0, -3::]
            # x_rgba[3] = mask
            x_rgba[0:3] = x_i[-3::]
            x_rgba[3] = mask
            x_rgba = x_rgba.permute(1, 2, 0).cpu().numpy()
            x_rgb.append(torch.Tensor(skimage.color.rgba2rgb(x_rgba)).permute(2, 0, 1).to(device))
        
        x_rgb = torch.stack(x_rgb)

        if self.use_thermal:
            x[:, -3::] = x_rgb
            return x
        else:
            return x_rgb