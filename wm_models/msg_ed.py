import torch.nn as nn
import torch
from .msg_diffuser import IndependentEncoder
from .msg_reverser import DeformableDecoder
from noise_layers.screen_shoot import ScreentoCameraDistortion
import torch.nn.functional as F
from utils import *


class Msg_ED(nn.Module):
    '''
    A Sequential of Encoder_MP-Noise-Decoder
    '''
    def __init__(self, message_length=64):
        super(Msg_ED, self).__init__()

        self.encoder = IndependentEncoder(message_length)
        self.noise = ScreentoCameraDistortion()
        self.decoder = DeformableDecoder(message_length)
    
    def forward(self, message, imgs, noise=True):
        Pw = self.encoder(message)

        Pw_r = F.interpolate(
            Pw,
            size=(imgs.size(2), imgs.size(3)),
            mode='bicubic',
            align_corners=False
        )

        Pw_r = F.avg_pool2d(Pw_r, kernel_size=3, stride=1, padding=1)

        Pw_r = torch.clamp(Pw_r, -0.012, 0.012)
        # Pw_r = torch.clamp(Pw_r, -1, 1)

        Pw_r[:, 1, :, :] = Pw_r[:, 1, :, :] * 0.2

        # # print(torch.min(Pw_r))

        imgs_w = imgs + Pw_r

        imgs_w = torch.clamp(imgs_w, -1, 1)

        if not noise:
            imgs_n = imgs_w
        
        else:
            imgs_n = self.noise([imgs_w, imgs_w])[0]

        # save_images_(imgs_n, './sim/', 1)
        # exit(0)
        
        imgs_n = torch.clamp(imgs_n, -1, 1)

        Ir, features = self.decoder(imgs_n)

        return Pw_r, imgs_w, features, Ir