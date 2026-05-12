import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

def random_float(min, max):
    """
    Return a random number
    :param min:
    :param max:
    :return:
    """
    return np.random.rand() * (max - min) + min


def get_random_rectangle_inside(image, height_ratio_range, width_ratio_range):
    """
    Returns a random rectangle inside the image, where the size is random and is controlled by height_ratio_range and width_ratio_range.
    This is analogous to a random crop. For example, if height_ratio_range is (0.7, 0.9), then a random number in that range will be chosen
    (say it is 0.75 for illustration), and the image will be cropped such that the remaining height equals 0.75. In fact,
    a random 'starting' position rs will be chosen from (0, 0.25), and the crop will start at rs and end at rs + 0.75. This ensures
    that we crop from top/bottom with equal probability.
    The same logic applies to the width of the image, where width_ratio_range controls the width crop range.
    :param image: The image we want to crop
    :param height_ratio_range: The range of remaining height ratio
    :param width_ratio_range:  The range of remaining width ratio.
    :return: "Cropped" rectange with width and height drawn randomly height_ratio_range and width_ratio_range
    """
    image_height = image.shape[2]
    image_width = image.shape[3]
    #np.rint是四舍五入取整，height_ratio_range[0]是为长变化的下限，[1]是为上限，
    # 所以是得到了一个缩小了的图像的长宽，在原图像的范围内
    remaining_height = int(np.rint(random_float(height_ratio_range[0], height_ratio_range[1]) * image_height))
    remaining_width = int(np.rint(random_float(width_ratio_range[0], width_ratio_range[0]) * image_width))

    if remaining_height == image_height:
        height_start = 0
    else:#np.random.randint()返回一个随机整数，包括低范围，不包括高范围
        height_start = np.random.randint(0, image_height - remaining_height)

    if remaining_width == image_width:
        width_start = 0
    else:
        width_start = np.random.randint(0, image_width - remaining_width)

    return height_start, height_start+remaining_height, width_start, width_start+remaining_width


class Crop(nn.Module):
    """
    Randomly crops the image from top/bottom and left/right. The amount to crop is controlled by parameters
    heigth_ratio_range and width_ratio_range
    """
    def __init__(self, height_ratio_range, width_ratio_range):
        """

        :param height_ratio_range:
        :param width_ratio_range:
        """
        super(Crop, self).__init__()
        self.height_ratio_range = np.sqrt(height_ratio_range)
        self.width_ratio_range = np.sqrt(width_ratio_range)


    def forward(self, noised_and_cover):
        n_a_c = [noised_and_cover[0].clone(), noised_and_cover[1].clone()]
        noised_image = n_a_c[0]
        # crop_rectangle is in form (from, to) where @from and @to are 2D points -- (height, width)

        h_start, h_end, w_start, w_end = get_random_rectangle_inside(noised_image, self.height_ratio_range, self.width_ratio_range)

        crop_mask = torch.zeros_like(noised_image)

        crop_mask[:, :, h_start:h_end, w_start:w_end] = 1

        n_a_c[0] = noised_image * crop_mask

        return n_a_c


class Crop_I(nn.Module):
    """
    Randomly crops the image and then interpolates the cropped region to the original size.
    """
    def __init__(self, height_ratio_range, width_ratio_range, interpolation_mode='bilinear'):
        super(Crop_I, self).__init__()
        self.height_ratio_range = height_ratio_range
        self.width_ratio_range = width_ratio_range
        self.interpolation_mode = interpolation_mode

    def forward(self, noised_and_cover):
        n_a_c = [noised_and_cover[0].clone(), noised_and_cover[1].clone()]
        noised_image = n_a_c[0]

        # Get the random crop rectangle
        h_start, h_end, w_start, w_end = get_random_rectangle_inside(
            noised_image, self.height_ratio_range, self.width_ratio_range
        )

        # Crop the image
        cropped_image = noised_image[:, :, h_start:h_end, w_start:w_end]

        # Interpolate the cropped image to the original size
        if cropped_image.shape[2] == 0 or cropped_image.shape[3] == 0:
            # If the cropped region is empty, return the original image
            return noised_and_cover

        # Use interpolation to resize the cropped image to the original size
        interpolated_image = F.interpolate(
            cropped_image,
            size=(noised_image.shape[2], noised_image.shape[3]),
            mode=self.interpolation_mode,
            align_corners=True if self.interpolation_mode == 'bilinear' else None
        )

        n_a_c[0] = interpolated_image

        return n_a_c


class Crop_Pad(nn.Module):
    """
    Randomly crops the image and then pads the cropped region to the original size.
    The padding can be either centered or placed in the top-right corner.
    """
    def __init__(self, height_ratio_range, width_ratio_range, padding_mode='center', padding_value=0):
        super(Crop_Pad, self).__init__()
        self.height_ratio_range = height_ratio_range
        self.width_ratio_range = width_ratio_range
        self.padding_mode = padding_mode
        self.padding_value = padding_value

    def forward(self, noised_and_cover):
        n_a_c = [noised_and_cover[0].clone(), noised_and_cover[1].clone()]
        noised_image = n_a_c[0]

        # Get the random crop rectangle
        h_start, h_end, w_start, w_end = get_random_rectangle_inside(
            noised_image, self.height_ratio_range, self.width_ratio_range
        )

        # Crop the image
        cropped_image = noised_image[:, :, h_start:h_end, w_start:w_end]

        # Calculate padding sizes
        original_height = noised_image.shape[2]
        original_width = noised_image.shape[3]
        cropped_height = cropped_image.shape[2]
        cropped_width = cropped_image.shape[3]

        # Calculate padding for height and width
        pad_height = original_height - cropped_height
        pad_width = original_width - cropped_width

        # Determine padding placement based on mode
        if self.padding_mode == 'center':
            # Center padding
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
        elif self.padding_mode == 'top_right':
            # Top-right padding (pad left and bottom)
            pad_top = 0
            pad_bottom = pad_height
            pad_left = pad_width
            pad_right = 0
        else:
            raise ValueError(f"Unsupported padding mode: {self.padding_mode}")

        # Apply padding to the cropped image
        padded_image = F.pad(
            cropped_image,
            (pad_left, pad_right, pad_top, pad_bottom),  # (left, right, top, bottom)
            mode='constant',
            value=self.padding_value
        )

        n_a_c[0] = padded_image

        return n_a_c


class Crop_(nn.Module):
    """
    Randomly crops the image and then interpolates the cropped region to the original size.
    """
    def __init__(self, height_ratio_range, width_ratio_range, interpolation_mode='bilinear'):
        super(Crop_, self).__init__()
        self.height_ratio_range = height_ratio_range
        self.width_ratio_range = width_ratio_range
        self.interpolation_mode = interpolation_mode

    def forward(self, noised_and_cover):
        n_a_c = [noised_and_cover[0].clone(), noised_and_cover[1].clone()]
        noised_image = n_a_c[0]

        _, _, h, w = noised_image.shape

        # Get the random crop rectangle
        h_start, h_end, w_start, w_end = get_random_rectangle_inside(
            noised_image, self.height_ratio_range, self.width_ratio_range
        )

        # Crop the image
        cropped_image = noised_image[:, :, h_start:h_end, w_start:w_end]

        # Interpolate the cropped image to the original size
        if cropped_image.shape[2] == 0 or cropped_image.shape[3] == 0:
            # If the cropped region is empty, return the original image
            return noised_and_cover
        
        imgs_resize = F.interpolate(
            cropped_image,
            size=(h, w),
            mode='bilinear',
            align_corners=False
        )

        n_a_c[0] = imgs_resize

        return n_a_c

# if __name__=='__main__':
#     x = torch.rand(8, 1, 128, 128)
#     c = Crop_Pad((0.04, 0.04), (0.04, 0.04))
#     x_, _ = c([x, x])

    