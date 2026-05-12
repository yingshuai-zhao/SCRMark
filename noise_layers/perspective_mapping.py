import torch
import torch.nn as nn
import numpy as np
import random
import kornia
import warnings

class Perspective_mapping(nn.Module):
    """
    Randomly crops the image and then interpolates the cropped region to the original size.
    """
    def __init__(self, d):
        super(Perspective_mapping, self).__init__()
        self.d = d

    def forward(self, noised_and_cover):
        n_a_c = [noised_and_cover[0].clone(), noised_and_cover[1].clone()]
        noised_image = n_a_c[0]

        _, _, h, w = noised_image.shape

        image_size = noised_image.shape[-1]
        batch_size = noised_image.shape[0]
        Ms = np.zeros((batch_size, 1, 3, 3))

        # 随机生成透视变换矩阵
        p_src = torch.ones(batch_size, 4, 2)
        p_dst = torch.ones(batch_size, 4, 2)
        for i in range(batch_size):
            x_lt = random.uniform(-self.d, self.d)    # 左上x坐标变换
            y_lt = random.uniform(-self.d, self.d)    # 左上y坐标变换
            x_lb = random.uniform(-self.d, self.d)    # 左下x坐标变换
            y_lb = random.uniform(-self.d, self.d)    # 左下y坐标变换
            x_rt = random.uniform(-self.d, self.d)    # 右上x坐标变换
            y_rt = random.uniform(-self.d, self.d)    # 右上y坐标变换
            x_rb = random.uniform(-self.d, self.d)    # 右下x坐标变换
            y_rb = random.uniform(-self.d, self.d)    # 右下y坐标变换

            p_src[i, :, :] = torch.tensor([
                [0, 0],
                [0, image_size],
                [image_size, 0],
                [image_size, image_size]])
                
            p_dst[i, :, :] = torch.tensor([
                [x_lt, y_lt],
                [x_lb, image_size + y_lb],
                [image_size + x_rt, y_rt],
                [image_size + x_rb, image_size + y_rb]])

        Ms = kornia.geometry.get_perspective_transform(p_src, p_dst).to(noised_image.device)

        n_a_c[0] = kornia.geometry.warp_perspective(noised_image.float(), Ms, dsize=(noised_image.shape[-2], noised_image.shape[-1])).to(noised_image.device)

        return n_a_c