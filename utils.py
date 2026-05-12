import os
import logging
from datetime import datetime
import torch
import numpy as np
import torch.nn.functional as F
import torchvision
from torch.nn.functional import mse_loss as mse


def save_valid_images(imgs, imgs_w, imgs_num, epoch, folder, modulation_map=None, resize_to=None):
    """
    保存第epoch轮验证过程第一个batch的前imgs_num张图片
    """
    images = imgs[:imgs_num, :, :, :].cpu()
    imgs_w = imgs_w[:imgs_num, :, :, :].cpu()

    # scale values to range [0, 1] from original range of [-1, 1]
    images = (images + 1) / 2
    imgs_w = (imgs_w + 1) / 2
    

    if resize_to is not None:
        images = F.interpolate(images, size=resize_to)
        imgs_w = F.interpolate(imgs_w, size=resize_to)
    imgs_residual = images - imgs_w

    if modulation_map != None:
        modulation_map = modulation_map[:imgs_num, :, :, :].cpu()
        modulation_map = modulation_map
        stacked_images = torch.cat([images, imgs_w, imgs_residual*10, modulation_map], dim=0)
    else:
        stacked_images = torch.cat([images, imgs_w, imgs_residual], dim=0)
    filename = os.path.join(folder, 'epoch-{}.png'.format(epoch))
    torchvision.utils.save_image(stacked_images, filename, normalize=False, nrow=imgs_num, padding=3, pad_value=1)

    
    
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


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(datetime.now().strftime('%y%m%d-%H%M%S')))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def psnr(input: torch.Tensor, target: torch.Tensor, max_val: float) -> torch.Tensor:
    input = clamp(((input.detach().cpu().squeeze()/2)+0.5) * max_val, 0, max_val)
    target = clamp(((target.detach().cpu().squeeze()/2)+0.5) * max_val, 0, max_val)
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor but got {type(target)}.")

    if not isinstance(target, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor but got {type(input)}.")

    if input.shape != target.shape:
        raise TypeError(f"Expected tensors of equal shapes, but got {input.shape} and {target.shape}")

    return 10.0 * torch.log10(max_val ** 2 / mse(input, target, reduction='mean'))


def alpha_fuser(image, watermark, alpha=5):
    """
    将水印信息与图像融合
    :param image: 输入图像，形状为 [3, H, W]，归一化到 [-1, 1]
    :param watermark: 水印信息，形状为 [3, H, W]，归一化到 [0, 1]
    :param alpha: 嵌入强度
    :return: 融合后的图像，归一化到 [-1, 1]
    """
    # 归一化图像到 [0, 1]
    image_denormalized = image * 0.5 + 0.5

    watermark_resized = F.interpolate(
            watermark,
            size=(image.shape[1], image.shape[2]),
            mode='bilinear',
            align_corners=False
        )
    
    # 确保水印在 [0, 1] 范围内
    watermark_resized = torch.clamp(watermark_resized, 0, 1)
    
    # 融合公式：Sw = α * Pw + (255 - α) * Sc
    fused_image = alpha * watermark_resized + (255 - alpha) * image_denormalized
    
    # 将融合后的图像归一化到 [0, 255] 范围内
    fused_image = torch.clamp(fused_image, 0, 255)
    
    # 重新归一化到 [-1, 1]
    fused_image_normalized = (fused_image / 255.0 - 0.5) / 0.5

    fused_image_resized = F.interpolate(
            fused_image_normalized.unsqueeze(0), 
            size=(watermark.shape[1], watermark.shape[2]),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
    return fused_image_resized, fused_image_normalized


def alpha_fuser_batch(batch_images, Pw, alpha=5):
    """
    将水印信息与图像融合，支持批量处理
    :param batch_images: 输入图像批量，形状为 [b, 3, H, W]，归一化到 [-1, 1]
    :param batch_watermarks: 水印信息批量，形状为 [b, 3, h, w]，归一化到 [0, 1]
    :param alpha: 嵌入强度
    :return: 融合后的图像批量，归一化到 [-1, 1]
    """
    # 归一化图像到 [0, 1]
    batch_images_denormalized = batch_images * 0.5 + 0.5
    
    # 对水印进行插值，使其与图像尺寸匹配

    if batch_images.shape == Pw.shape:
        batch_watermarks_resized = Pw
    else:
        batch_watermarks_resized = F.interpolate(
            Pw,
            size=(batch_images.size(2), batch_images.size(3)),
            mode='bilinear',
            align_corners=False
        )
    
    # 确保水印在 [0, 1] 范围内
    batch_watermarks_resized = torch.clamp(batch_watermarks_resized, 0, 1)
    
    # 融合公式：Sw = α * Pw + (255 - α) * Sc
    fused_images = alpha * batch_watermarks_resized + (255 - alpha) * batch_images_denormalized
    
    # 将融合后的图像归一化到 [0, 255] 范围内
    fused_images = torch.clamp(fused_images, 0, 255)
    
    # 重新归一化到 [-1, 1]
    fused_images_normalized = (fused_images / 255.0 - 0.5) / 0.5
    
    return fused_images_normalized


def setup_seed(seed):
    """
    set random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def decoded_message_error_rate_bit(message, decoded_message):
    length = message.shape[0]
    message = message.gt(0.5)
    decoded_message = decoded_message.gt(0.5)
    error_rate = float(sum(message != decoded_message)) / length
    return error_rate


def decoded_message_error_rate_bit_batch(messages, decoded_messages, mode='mean'):

    decoded_messages_mean = torch.mean(decoded_messages, dim=0, keepdim=True)

    messages_ori = messages

    if messages.shape[0] == 1:
        messages = messages.repeat((decoded_messages.shape[0], 1))
    elif messages.shape[0] == decoded_messages.shape[0]:
        pass
    else:
        print('messages size not match, messages:{}, decoded_messages:{}'.format(messages.shape, decoded_messages.shape))

    batch_size = len(messages)
    error_rate = []
    for i in range(batch_size):
        err = decoded_message_error_rate_bit(messages[i], decoded_messages[i])
        error_rate.append(err)
    if mode == 'mean':
        error_rate = decoded_message_error_rate_bit(messages_ori[0], decoded_messages_mean[0])
    elif mode == 'min':
        error_rate = np.min(error_rate)
    elif mode == 'fusion':
        error_rate = decoded_message_error_rate_bit(messages_ori[0], torch.nn.Sigmoid()(decoded_messages[0]))
    else:
        print('decoded_message_error_rate_bit_batch: mode:{} is not right'.format(mode))
        return
    return error_rate


def bit_err_count(message, target_message):
    message1 = message.gt(0.5)
    target_message1 = target_message.gt(0.5)
    return torch.sum(torch.logical_xor(message1, target_message1), dim=-1)


def get_similar_message_list(message, threshold=5):
    batch_size = message.shape[0]
    candidate = []
    for i in range(batch_size):
        # obtain similar message within given threshold
        simlist_tmp = []
        for j in range(batch_size):
            err = bit_err_count(message[i:i+1],message[j:j+1])
            if err <= threshold:
                simlist_tmp.append(message[j:j+1])

        # insert the mean of simlist into candidate, and sort candidate according to the size of simlist
        if len(simlist_tmp) >= 2:
            t = torch.vstack(simlist_tmp)
            if len(candidate) == 0:
                candidate.append([len(simlist_tmp), torch.mean(t.float(), dim=0, keepdim=True)])
            else:
                ll = len(candidate)
                for n in range(ll):
                    if candidate[n][0] < len(simlist_tmp):
                        candidate.insert(n, [len(simlist_tmp), torch.mean(t.float(), dim=0, keepdim=True)])
    if len(candidate) != 0:
        result = [a[1] for a in candidate]
    else:
        result = []

    return result


def messgae_fusion(messages, depth=2):
    """
    return a final result based on the consistency among messages
    """
    if messages.shape[0] == 1:
        return messages
    if messages.shape[0] == 2:
        final = torch.mean(messages.float(), dim=0, keepdim=True)
        return final

    final = 0
    for i in range(5):
        candidate = get_similar_message_list(messages, threshold=i)
        if len(candidate) == 0:
            continue
        elif len(candidate) == 1:
            final = candidate[0]
        elif depth > 1:
            final = messgae_fusion(torch.vstack(candidate), depth=1)
        else:
            final = candidate[0]
        return final

    if isinstance(final, int):
        final = torch.mean(messages.float(), dim=0, keepdim=True)

    return final