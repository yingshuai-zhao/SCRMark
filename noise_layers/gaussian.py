import math

import utils
import torch
import torch.nn as nn
import torch.nn.functional as F


class Gaussian_blur(nn.Module):
    def __init__(self,kernel,sigma):
        super(Gaussian_blur, self).__init__()
        self.kernel=int(kernel)
        self.sigma=float(sigma)


    def forward(self, noised_and_cover):
        # get the gaussian filter
        n_a_c = [noised_and_cover[0].clone(), noised_and_cover[1].clone()]
        encode_image=n_a_c[0]
        batch_size,channel=encode_image.shape[0],encode_image.shape[1]
        assert encode_image.shape[1]==3|1
        gaussian_kernel=utils.gaussian_kernel(self.sigma,self.kernel)
        kernel = torch.FloatTensor(gaussian_kernel).unsqueeze(0).unsqueeze(0)
        kernel=kernel.expand(channel,1,self.kernel,self.kernel).to(encode_image.device)
        # print(kernel)
        # print(kernel.size())
        # print(encode_image.size())
        # weight = nn.Parameter(data=kernel, requires_grad=False)
        n_a_c[0]=F.conv2d(encode_image,kernel,stride=1,padding=1,groups=3)
        return n_a_c



        