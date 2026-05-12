import torch
import torch.nn as nn

class Gaussian_Noise(nn.Module):
    def __init__(self, mean, sigma):
        super(Gaussian_Noise, self).__init__()
        self.mean = float(mean)
        self.sigma = float(sigma)

    def forward(self, noise_and_cover):
        n_a_c = [noise_and_cover[0].clone(), noise_and_cover[1].clone()]
        encode_image = n_a_c[0]
        B, C, H, W = encode_image.size()
        
        # 使用 torch.normal 生成高斯噪声
        noise = torch.normal(mean=self.mean, std=self.sigma, size=(B, 1, H, W), device=encode_image.device)
        # 使用 torch.clamp 限制噪声范围
        noise = torch.clamp(noise, 0, 1)
        
        # 将噪声调整为与 encode_image 的每个通道匹配的形状
        noise = noise.expand(B, C, H, W)  # 扩展噪声到 (B, C, H, W)
        noise = noise * 0.25
        
        # 将噪声添加到 encode_image
        encode_image = encode_image + noise
        
        n_a_c[0] = encode_image
        return n_a_c