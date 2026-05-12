import numpy as np
import torch.nn as nn
from .diff_jpeg.jpeg import DiffJPEGCoding_
from .jpeg_tri import JpegSS_, JpegTest_
from .Gaussian_noise import Gaussian_Noise
from .cropout import Cropout
from .crop import Crop, Crop_I, Crop_Pad, Crop_
from .resize import Resize
from .identity import Identity
from .padding import Padding
from .perspective_mapping import Perspective_mapping
from .screen_shoot import ScreentoCameraDistortion

class Noiser(nn.Module): 
    """
    This module allows to combine different noise layers into a sequential noise module. The
    configuration and the sequence of the noise layers is controlled by the noise_config parameter.
    """
    def __init__(self, mode='IL'):
        super(Noiser, self).__init__()
        if mode == 'IL':
            self.noise_layers = [Identity(), Crop_([0.9, 1], [0.9, 1]), Resize([0.6, 2]), Padding(0, 5)]
        # elif mode == 'PL':
        #     self.noise_layers = [Identity(), Gaussian_Noise(0, 1), DiffJPEGCoding_(70, ste=True)]
        # else:
        #     self.noise_layers = [Identity(), Crop([0.7, 1], [0.7, 1]), Cropout([0, 0.2], [0, 0.2]), Resize([0.8, 3]), Gaussian_Noise(0, 1), DiffJPEGCoding_(70, ste=True)]
        elif mode == 'PL':
            self.noise_layers = [Identity(), Gaussian_Noise(0, 1), JpegSS_(75)]
        elif mode == 'AllMix':
            self.noise_layers = [Identity(), Crop_([0.9, 1], [0.9, 1]), Perspective_mapping(4), Resize([0.6, 2]), Gaussian_Noise(0, 1), DiffJPEGCoding_(50, ste=True), ScreentoCameraDistortion()]
        else:
            exit(1)
        


    def forward(self, encoded_and_cover, mode='train'):
        if mode == 'train':
            p = [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3]
            random_noise_layer = np.random.choice(self.noise_layers, 1, p=p)[0]
            # random_noise_layer = np.random.choice(self.noise_layers, 1)[0]
        elif mode == 'valid':
            random_noise_layer = ScreentoCameraDistortion()

        return random_noise_layer(encoded_and_cover)

