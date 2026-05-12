import torch
import torch.nn as nn
import kornia
import random
import numpy as np
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as F
from .diff_jpeg.jpeg import DiffJPEGCoding
import warnings
warnings.filterwarnings("ignore")

class ScreentoCameraDistortion(nn.Module):
    def __init__(self, mode='IL'):
        """
        屏幕拍摄失真模拟层
        """
        super(ScreentoCameraDistortion, self).__init__()
        self.jpeg_compression = DiffJPEGCoding(80, ste=True)
        self.mode = mode
    
    def forward(self, noise_and_cover):

        n_a_c = [noise_and_cover[0].clone(), noise_and_cover[1].clone()]
        watermarked_img = n_a_c[0]

        orig_size = watermarked_img.shape[-2:]

        if self.mode == 'IL':

            watermarked_img = self.random_resize(watermarked_img)

            watermarked_img = self.random_crop(watermarked_img)

            watermarked_img = self.perspective_distortion(watermarked_img)
        
        else:
            watermarked_img = self.color_distortion(watermarked_img)

            watermarked_img = self.gaussian_blur(watermarked_img)

            watermarked_img = self.random_resize(watermarked_img)

            watermarked_img = self.perspective_distortion(watermarked_img)

            watermarked_img += self.light_distortion(watermarked_img)

            watermarked_img += self.moire_distortion(watermarked_img)*0.3

            watermarked_img = self.jpeg_compression(watermarked_img)
        
        watermarked_img = F.resize(watermarked_img, orig_size)

        n_a_c[0] = watermarked_img.clamp(-1, 1)
        
        return n_a_c

    def gaussian_blur(self, img):
        kernel_size = random.choice([3, 5, 7])
        sigma = random.uniform(0.5, 2.0)
        return kornia.filters.gaussian_blur2d(img, (kernel_size, kernel_size), (sigma, sigma))

    def perspective_distortion(self, img):
        batch_size, _, H, W = img.shape
        d = max(H, W) * 0.05  # Up to 10%
        
        # Randomly generate a perspective transformation matrix
        p_src = torch.ones(batch_size, 4, 2)
        p_dst = torch.ones(batch_size, 4, 2)
        for i in range(batch_size):
            x_lt = random.uniform(-d, d)  
            y_lt = random.uniform(-d, d)   
            x_lb = random.uniform(-d, d)   
            y_lb = random.uniform(-d, d)   
            x_rt = random.uniform(-d, d)   
            y_rt = random.uniform(-d, d)   
            x_rb = random.uniform(-d, d)   
            y_rb = random.uniform(-d, d)   

            p_src[i, :, :] = torch.tensor([
                [0, 0],
                [0, H],
                [W, 0],
                [W, H]])
                
            p_dst[i, :, :] = torch.tensor([
                [x_lt, y_lt],
                [x_lb, H + y_lb],
                [W + x_rt, y_rt],
                [W + x_rb, H + y_rb]])

        Ms = kornia.geometry.get_perspective_transform(p_src, p_dst).to(img.device)

        return kornia.geometry.warp_perspective(img.float(), Ms, dsize=(img.shape[-2], img.shape[-1])).to(img.device)


    def light_distortion(self, imgs):
        img_size = imgs.shape[-1]
        mask = np.zeros((imgs.shape))
        x = np.random.randint(0,mask.shape[2])
        y = np.random.randint(0,mask.shape[3])
        max_len = np.max([np.sqrt(x**2+y**2),np.sqrt((x-img_size)**2+y**2),np.sqrt(x**2+(y-img_size)**2),np.sqrt((x-img_size)**2+(y-img_size)**2)])/2
        for i in range(mask.shape[2]):
            for j in range(mask.shape[3]):
                l = np.sqrt((i-x)**2+(j-y)**2)
                if l < max_len:
                    mask[:,:,i,j] = 1 - l/max_len
        O = mask
        return torch.from_numpy(O.copy()).to(imgs.device)

    def color_distortion(self, img):
        if img.min() < 0:
            imgs_normalized = (img + 1) / 2
        else:
            imgs_normalized = img.clone()

        brightness = random.uniform(0.5, 1.5)
        contrast = random.uniform(0.5, 1.5)
        saturation = random.uniform(0.5, 1.5)
        
        adjusted_imgs = kornia.enhance.adjust_brightness(imgs_normalized, brightness) 
        adjusted_imgs = kornia.enhance.adjust_contrast(adjusted_imgs, contrast) 
        adjusted_imgs = kornia.enhance.adjust_saturation(adjusted_imgs, saturation) 

        if img.min() < 0:
            adjusted_imgs = adjusted_imgs * 2 - 1
        
        return img

    def moire_distortion(self, imgs):
        masks = torch.zeros_like(imgs).to(imgs.device)

        for i in range(3):
            theta = torch.randint(0,180,(1,)).to(imgs.device)
            center_x = torch.rand(1)*imgs.shape[-2]
            center_y = torch.rand(1)*imgs.shape[-1]
            
            x = torch.arange(imgs.shape[-2], dtype=torch.float)
            y = torch.arange(imgs.shape[-1], dtype=torch.float)
            grid_x, grid_y = torch.meshgrid(x, y)

            dist_x = grid_x - center_x
            dist_y = grid_y - center_y

            grid_x, grid_y = grid_x.to(imgs.device), grid_y.to(imgs.device)
            dist_x, dist_y = dist_x.to(imgs.device), dist_y.to(imgs.device)

            z1 = 0.5 + 0.5*torch.cos(2*3.14159*torch.sqrt(torch.square(dist_x) + torch.square(dist_y)))
            z2 = 0.5 + 0.5*torch.cos(torch.cos(theta/180*3.14159)*grid_y + torch.sin(theta/180*3.14159)*grid_x)
            mask = torch.min(z1, z2)

            masks[:,i,:,:] = mask

        masks = (masks*2)-1

        return masks

    def random_resize(self, img):
        scale_factor = random.uniform(0.5, 2.0)
        *_, H, W = img.shape
        new_size = (int(H * scale_factor), int(W * scale_factor))
        return F.resize(img, new_size)

    def random_crop(self, img):
        *_, H, W = img.shape
        
        crop_ratio_h = random.uniform(0.9, 1.0)
        crop_ratio_w = random.uniform(0.9, 1.0)

        crop_h = int(H * crop_ratio_h)
        crop_w = int(W * crop_ratio_w)

        top = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)
        
        return F.crop(img, top, left, crop_h, crop_w)