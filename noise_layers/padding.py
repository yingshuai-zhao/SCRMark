import torch
import torch.nn as nn
import random
import torch.nn.functional as F

def random_padding_numbers(p_min, p_max):
    # 确保p_min小于等于p_max
    if p_min > p_max:
        p_min, p_max = p_max, p_min
    
    # 从p_min到p_max之间随机选择4个数（允许重复）
    numbers = tuple(random.randint(p_min, p_max) for _ in range(4))
    
    # 返回结果
    return numbers

class Padding(nn.Module):
    def __init__(self, p_min, p_max):
        super(Padding, self).__init__()
        self.p_min = p_min
        self.p_max = p_max

    def forward(self, noise_and_cover):
        n_a_c = [noise_and_cover[0].clone(), noise_and_cover[1].clone()]
        encode_image = n_a_c[0]

        _, _, h, w = encode_image.shape
        
        pad_nums = random_padding_numbers(self.p_min, self.p_max)

        noise_image = torch.nn.functional.pad(encode_image, pad_nums, mode='replicate')

        imgs_resize = F.interpolate(
            noise_image,
            size=(h, w),
            mode='bilinear',
            align_corners=False
        )

        n_a_c[0] = imgs_resize
        
        return n_a_c

