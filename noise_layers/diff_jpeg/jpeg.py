from typing import Optional

import torch.nn as nn
from torch import Tensor

from .encode import jpeg_encode
from .decode import jpeg_decode
from .rounding import *
from .clipping import *
from .utils import QUANTIZATION_TABLE_C, QUANTIZATION_TABLE_Y


def clamp(input, min=None, max=None):
    ndim = input.ndimension()
    if min is None:
        pass
    elif isinstance(min, (float, int)):
        input = torch.clamp(input, min=min)
    elif isinstance(min, torch.Tensor):
        if min.ndimension() == ndim - 1 and min.shape == input.shape[1:]:
            input = torch.max(input, min.view(1, *min.shape))
        else:
            assert min.shape == input.shape
            input = torch.max(input, min)
    else:
        raise ValueError("min can only be None | float | torch.Tensor")

    if max is None:
        pass
    elif isinstance(max, (float, int)):
        input = torch.clamp(input, max=max)
    elif isinstance(max, torch.Tensor):
        if max.ndimension() == ndim - 1 and max.shape == input.shape[1:]:
            input = torch.min(input, max.view(1, *max.shape))
        else:
            assert max.shape == input.shape
            input = torch.min(input, max)
    else:
        raise ValueError("max can only be None | float | torch.Tensor")
    return input


def diff_jpeg_coding(
    image_rgb: Tensor,
    jpeg_quality: Tensor,
    quantization_table_y: Optional[Tensor] = None,
    quantization_table_c: Optional[Tensor] = None,
    ste: bool = False,
) -> Tensor:
    """Performs JPEG encoding.

    Args:
        image_rgb (Tensor): RGB input images of the shape [B, 3, H, W].
        jpeg_quality (Tensor): Compression strength of the shape [B].
        quantization_table_y (Tensor): Quantization table Y channel shape [8, 8]. Default None (default QT is used).
        quantization_table_c (Tensor): Quantization table C channels shape [8, 8]. Default None (default QT is used).
        ste (bool): If true STE version of differentiable rounding, floor, and clipping is used. Default False.

    Returns:
        image_rgb_jpeg (Tensor): JPEG-coded image of the shape [B, 3, H, W].
    """
    # Get QT if not given
    if quantization_table_y is None:
        quantization_table_y: Tensor = QUANTIZATION_TABLE_Y
    if quantization_table_c is None:
        quantization_table_c: Tensor = QUANTIZATION_TABLE_C
    # Get rounding, floor, and clipping functions
    if ste:
        rounding_function = differentiable_rounding_ste
        floor_function = differentiable_floor_ste
        clipping_function = differentiable_clipping_ste
    else:
        clipping_function = differentiable_clipping
        rounding_function = differentiable_polynomial_rounding
        floor_function = differentiable_polynomial_floor
    # Get original shape
    _, _, H, W = image_rgb.shape  
    # Perform encoding
    y_encoded, cb_encoded, cr_encoded = jpeg_encode(
        image_rgb=image_rgb,
        jpeg_quality=jpeg_quality,
        quantization_table_c=quantization_table_c,
        quantization_table_y=quantization_table_y,
        rounding_function=rounding_function,
        floor_function=floor_function,
        clipping_function=clipping_function,
    )  
    image_rgb_jpeg: Tensor = jpeg_decode(
        input_y=y_encoded,
        input_cb=cb_encoded,
        input_cr=cr_encoded,
        jpeg_quality=jpeg_quality,
        H=H,
        W=W,
        quantization_table_c=quantization_table_c,
        quantization_table_y=quantization_table_y,
        floor_function=floor_function,
        clipping_function=clipping_function,
    )
    # Clip coded image
    image_rgb_jpeg = clipping_function(image_rgb_jpeg, 0, 255)
    return image_rgb_jpeg


class DiffJPEGCoding(nn.Module):
    """This class implements our differentiable JPEG coding."""

    def __init__(self, factor, ste: bool = False) -> None:
        """Constructor method.

        Args:
            ste (bool): If true STE version of differentiable rounding, floor, and clipping is used. Default False.
        """
        # Call super constructor
        super(DiffJPEGCoding, self).__init__()
        # Save parameter
        self.ste: bool = ste
        self.factor = float(factor)

    def forward(
        self,
        image_rgb: Tensor,
        quantization_table_y: Optional[Tensor] = None,
        quantization_table_c: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass performs JPEG coding.

        Args:
            image_rgb (Tensor): RGB input images of the shape [B, 3, H, W].
            jpeg_quality (Tensor): Compression strength of the shape [B].
            quantization_table_y (Tensor): Quantization table Y channel shape [8, 8]. Default None (default QT is used).
            quantization_table_c (Tensor): Quantization table C channels shape [8, 8]. Default None (default QT is used).

        Returns:
            image_rgb_jpeg (Tensor): JPEG coded image of the shape [B, 3, H, W].
        """
        image_rgb = (clamp(image_rgb, -1, 1) + 1) * 255 / 2

        jpeg_quality = (torch.ones(image_rgb.shape[0], dtype=torch.float) * self.factor).to(image_rgb.device)
        # Perform coding
        image_rgb_jpeg: Tensor = diff_jpeg_coding(
            image_rgb=image_rgb,
            jpeg_quality=jpeg_quality,
            quantization_table_c=quantization_table_c,
            quantization_table_y=quantization_table_y,
            ste=self.ste,
        )
        image_rgb_jpeg = image_rgb_jpeg * 2 / 255 - 1

        return clamp(image_rgb_jpeg, -1, 1)
 
class DiffJPEGCoding_(nn.Module):
    """This class implements our differentiable JPEG coding."""

    def __init__(self, factor, ste: bool = False) -> None:
        """Constructor method.

        Args:
            ste (bool): If true STE version of differentiable rounding, floor, and clipping is used. Default False.
        """
        # Call super constructor
        super(DiffJPEGCoding_, self).__init__()
        # Save parameter
        self.ste: bool = ste
        self.factor = float(factor)

    def forward(
        self,
        image_rgb_list: Tensor,
        quantization_table_y: Optional[Tensor] = None,
        quantization_table_c: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass performs JPEG coding.

        Args:
            image_rgb (Tensor): RGB input images of the shape [B, 3, H, W].
            jpeg_quality (Tensor): Compression strength of the shape [B].
            quantization_table_y (Tensor): Quantization table Y channel shape [8, 8]. Default None (default QT is used).
            quantization_table_c (Tensor): Quantization table C channels shape [8, 8]. Default None (default QT is used).

        Returns:
            image_rgb_jpeg (Tensor): JPEG coded image of the shape [B, 3, H, W].
        """
        image_rgb = (clamp(image_rgb_list[0], -1, 1) + 1) * 255 / 2

        jpeg_quality = (torch.ones(image_rgb.shape[0], dtype=torch.float) * self.factor).to(image_rgb.device)
        # Perform coding
        image_rgb_jpeg: Tensor = diff_jpeg_coding(
            image_rgb=image_rgb,
            jpeg_quality=jpeg_quality,
            quantization_table_c=quantization_table_c,
            quantization_table_y=quantization_table_y,
            ste=self.ste,
        )
        image_rgb_jpeg = image_rgb_jpeg * 2 / 255 - 1

        image_rgb_list[0] = clamp(image_rgb_jpeg, -1, 1)

        return image_rgb_list