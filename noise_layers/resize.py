import torch.nn as nn
import torch.nn.functional as F
from .crop import random_float

class Resize(nn.Module):
    """
    Resize the image.
    """
    def __init__(self, opt):
        super(Resize, self).__init__()
        self.resize_ratio_down = random_float(opt[0], opt[1])
        self.interpolation_method = 'nearest'

    def forward(self, noised_and_cover):
        n_a_c = [noised_and_cover[0].clone(), noised_and_cover[1].clone()]

        wm_imgs = n_a_c[0]

        h, w = wm_imgs.shape[2], wm_imgs.shape[3]
        scaled_h = int(self.resize_ratio_down * h)
        scaled_w = int(self.resize_ratio_down * w)
        #
        noised_down = F.interpolate(
                                    wm_imgs,
                                    size=(scaled_h, scaled_w),
                                    mode=self.interpolation_method
                                    )
        n_a_c[0] = F.interpolate(
                                    noised_down,
                                    size=(h, w),
                                    mode=self.interpolation_method
                                    )

        return n_a_c
