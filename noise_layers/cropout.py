import torch
import torch.nn as nn
import numpy as np
from .crop import get_random_rectangle_inside

class Cropout(nn.Module):
    """
    Combines the noised and cover images into a single image, as follows: Takes a crop of the noised image, and takes the rest from
    the cover image. The resulting image has the same size as the original and the noised images.
    """
    def __init__(self, height_ratio_range, width_ratio_range):
        super(Cropout, self).__init__()
        self.height_ratio_range = np.sqrt(height_ratio_range)
        self.width_ratio_range = np.sqrt(width_ratio_range)

    def forward(self, noised_and_cover):
        n_a_c = [noised_and_cover[0].clone(), noised_and_cover[1].clone()]
        noised_image = n_a_c[0]
        # crop_rectangle is in form (from, to) where @from and @to are 2D points -- (height, width)

        h_start, h_end, w_start, w_end = get_random_rectangle_inside(noised_image, self.height_ratio_range, self.width_ratio_range)

        cropout_mask = torch.ones_like(noised_image)

        cropout_mask[:, :, h_start:h_end, w_start:w_end] = 0

        n_a_c[0] = noised_image * cropout_mask

        return n_a_c