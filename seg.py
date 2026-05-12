import torch
from torchvision import transforms
import numpy as np
import cv2
import torch.nn.functional as F
from math import *
from torchvision import utils as vutils


interpolatemode = 'bicubic'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


def generate_mask(image, model):
    """
    return mask
    """
    if isinstance(image, list):
        image = [F.interpolate(im, (512,512), mode=interpolatemode) for im in image]
        image = torch.vstack(image)
    else:
        image = F.interpolate(image, (512,512), mode=interpolatemode)
    with torch.no_grad():
        image = image
        d0, d1, d2, d3, d4, d5, d6 = model(image)
    return d0



def rectify(img, mask, threshold=128, use_gpu_perspective=True):
    """
    rectify geometric distoration with perspective transform
    """

    def order_points(pts):
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def perspective_crop_cv(img, cnt, dst_size):
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            src_pts = approx.reshape(4, 2).astype(np.float32)
        else:
            rect = cv2.minAreaRect(cnt)
            src_pts = cv2.boxPoints(rect).astype(np.float32)
        src_pts = order_points(src_pts)
        dst_pts = np.array([[0, 0],
                            [dst_size[0]-1, 0],
                            [dst_size[0]-1, dst_size[1]-1],
                            [0, dst_size[1]-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, M, dst_size)
        return warped

    if use_gpu_perspective:
        try:
            import kornia.geometry.transform as K
            import torch.nn.functional as F
            _kornia_available = True
        except ImportError:
            print("Warning: kornia not installed, fallback to CPU perspective transform")
            _kornia_available = False
    else:
        _kornia_available = False

    def perspective_crop_gpu(img_np, cnt, dst_size, device):
        # img_np: HWC uint8, cnt: contour, dst_size: (w, h)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            src_pts = approx.reshape(4, 2).astype(np.float32)
        else:
            rect = cv2.minAreaRect(cnt)
            src_pts = cv2.boxPoints(rect).astype(np.float32)
        src_pts = order_points(src_pts)
        dst_pts = np.array([[0, 0],
                            [dst_size[0]-1, 0],
                            [dst_size[0]-1, dst_size[1]-1],
                            [0, dst_size[1]-1]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts).astype(np.float32)

        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

        M_tensor = torch.from_numpy(M).unsqueeze(0).to(device)
        warped = K.warp_perspective(img_tensor, M_tensor, dsize=(dst_size[1], dst_size[0]))  # dsize = (h, w)
        warped = (warped.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return warped


    if isinstance(img, torch.Tensor):
        image = img.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
        image = (image + 1.) * 127.5
    else:
        image = np.array(img)

    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask_bin = np.uint8(mask * 255)
    if threshold != 128:
        mask_bin = (mask_bin >= threshold).astype(np.uint8) * 255
    else:
        mask_bin[mask_bin < threshold] = 0
        mask_bin[mask_bin >= threshold] = 255


    _, thresh = cv2.threshold(mask_bin, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]


    angles = [0]
    for cnt in sorted_contours:
        rec = cv2.minAreaRect(cnt)
        width, height = rec[1]
        angle = rec[2]
        if width >= 55 and height >= 55 and width <= 4 * height and height <= 4 * width:
            if abs(angle) > 45:
                if angle < 0:
                    angle = -(angle + 90)
                else:
                    angle = 90 - angle
            else:
                angle = -angle
            angle = round(angle)
            angles.append(angle)

    angles = search_near(angles, [16, 12, 8, 4, 2])
    angle = np.mean(angles)


    M_rot = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), -angle, 1)
    rotate_img = cv2.warpAffine(image, M_rot, (image.shape[1], image.shape[0]))
    rotate_mask = cv2.warpAffine(mask_bin, M_rot, (mask_bin.shape[1], mask_bin.shape[0]))


    rotate_mask[rotate_mask < threshold] = 0
    rotate_mask[rotate_mask >= threshold] = 255
    _, thresh2 = cv2.threshold(rotate_mask, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)[:20]


    MIN_BLOCK_SIZE = 200
    ASPECT_RATIO_MAX = 2.0
    SIZE_LOWER_RATIO = 0.6
    SIZE_UPPER_RATIO = 1.4
    SOLIDITY_THRESH = 0.6

    candidate_info = []      # (cnt, x, y, w, h, rect_area, contour_area)
    heights = []
    widths = []
    for cnt in sorted_contours2:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= MIN_BLOCK_SIZE and w >= MIN_BLOCK_SIZE and w <= ASPECT_RATIO_MAX * h and h <= ASPECT_RATIO_MAX * w:
            rect_area = w * h
            cnt_area = cv2.contourArea(cnt)
            candidate_info.append((cnt, x, y, w, h, rect_area, cnt_area))
            heights.append(h)
            widths.append(w)


    if heights:
        height_cluster = search_near(heights, [384, 256, 128, 64, 32])
        width_cluster = search_near(widths, [384, 256, 128, 64, 32])
        h_ref = np.mean(height_cluster) if height_cluster else np.mean(heights)
        w_ref = np.mean(width_cluster) if width_cluster else np.mean(widths)
    else:
        h_ref = w_ref = 0


    accepted_blocks = []   # (cnt, x, y, w, h, target_w, target_h)
    for cnt, x, y, w, h, rect_area, cnt_area in candidate_info:

        if h_ref > 0 and w_ref > 0:
            hr = h / h_ref
            wr = w / w_ref
            if not (SIZE_LOWER_RATIO <= hr <= SIZE_UPPER_RATIO and SIZE_LOWER_RATIO <= wr <= SIZE_UPPER_RATIO):
                continue

        if rect_area > 0 and cnt_area / rect_area < SOLIDITY_THRESH:
            continue

        rect = cv2.minAreaRect(cnt)
        raw_w, raw_h = rect[1]
        target_w = max(int(round(raw_w)), 32)
        target_h = max(int(round(raw_h)), 32)
        accepted_blocks.append((cnt, x, y, w, h, target_w, target_h))


    if not accepted_blocks:
        if candidate_info:

            for cnt, x, y, w, h, _, _ in candidate_info:
                rect = cv2.minAreaRect(cnt)
                raw_w, raw_h = rect[1]
                target_w = max(int(round(raw_w)), 32)
                target_h = max(int(round(raw_h)), 32)
                accepted_blocks.append((cnt, x, y, w, h, target_w, target_h))
        else:

            accepted_blocks = [(None, 0, 0, rotate_img.shape[1], rotate_img.shape[0],
                                rotate_img.shape[1], rotate_img.shape[0])]


    final_mask = np.zeros_like(rotate_mask, dtype=np.uint8)
    images_list = []
    height_list = []
    width_list = []

    if use_gpu_perspective and _kornia_available:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        perspective_func = lambda img_np, cnt, dst_size: perspective_crop_gpu(img_np, cnt, dst_size, device)
    else:
        perspective_func = perspective_crop_cv

    for (cnt, x, y, w, h, target_w, target_h) in accepted_blocks:

        final_mask[y:y+h, x:x+w] = 255

        try:
            if cnt is None:  
                block = rotate_img[y:y+h, x:x+w]
                block_resized = cv2.resize(block, (256, 256))
            else:

                block_resized = perspective_func(rotate_img, cnt, (256, 256))
        except Exception:

            block = rotate_img[y:y+h, x:x+w]
            block_resized = cv2.resize(block, (256, 256))
        images_list.append(block_resized)
        height_list.append(target_h)
        width_list.append(target_w)

    if not height_list:
        height_list.append(128)
    if not width_list:
        width_list.append(128)
    height_list = search_near(height_list, [384, 256, 128, 64, 32])
    width_list = search_near(width_list, [384, 256, 128, 64, 32])


    images_list = [torch.tensor(im) for im in images_list]
    images_tensor = torch.stack(images_list).permute(0, 3, 1, 2).float()
    images_tensor = (images_tensor / 255.0 - 0.5) / 0.5

    return images_tensor, [np.mean(height_list), np.mean(width_list)], angle


def search_near(num_list, qs_list=[]):
    """
    search near results within given quantization step
    """
    for qs in qs_list:
        num_list_bak = []
        for i in num_list:
            num_list_bak.append(i // qs * qs)
        maxlabel = max(set(num_list_bak), key=num_list_bak.count)
        result = []
        for i in range(len(num_list_bak)):
            if num_list_bak[i] == maxlabel:
                result.append(num_list[i])
        num_list = result
    return num_list


def pad_split_seg_rectify_multi(
    image,
    model,
    name,
    targetH=512,
    targetW=512,
    base_patch_size=512,
    min_overlap=64,
    patch_size_list=[512, 640, 768, 896, 1024],
    fusion_mode='vote',
    vote_thresh=0.5,
    batch_patch_num=20
):
    """
    Multi-scale segment and rectify with sliding window approach.
    """

    batch, channels, height, width = image.shape
    device = image.device

    # --------- 尺度配置 -----------
    if patch_size_list is None:
        patch_size_list = [base_patch_size]
    else:
        if isinstance(patch_size_list, int):
            patch_size_list = [patch_size_list]

    def create_gaussian_weights(size, sigma=0.3):
        """Create 2D Gaussian weights for blending"""
        center = (size - 1) / 2
        x = torch.arange(size, dtype=torch.float32, device=device)
        y = torch.arange(size, dtype=torch.float32, device=device)
        x, y = torch.meshgrid(x, y, indexing='ij')
        weights = torch.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * (sigma * size) ** 2))
        return weights.unsqueeze(0).unsqueeze(0)

    def _seg_single_scale(patch_size):
        overlap = min(min_overlap, patch_size // 8)
        stride = patch_size - overlap

        padH = (stride - (height - patch_size) % stride) % stride if height > patch_size else 0
        padW = (stride - (width - patch_size) % stride) % stride if width > patch_size else 0

        image_pad = F.pad(image, (0, padW, 0, padH), mode='constant', value=0)
        padded_height, padded_width = height + padH, width + padW

        weight = create_gaussian_weights(patch_size)

        mask_accum = torch.zeros((batch, 1, padded_height, padded_width), device=device)
        weight_accum = torch.zeros((1, 1, padded_height, padded_width), device=device)

        num_patches_h = max(1, (padded_height - patch_size) // stride + 1)
        num_patches_w = max(1, (padded_width - patch_size) // stride + 1)

        all_patches = []
        positions = []

        for i in range(num_patches_h):
            for j in range(num_patches_w):
                h_start = i * stride
                w_start = j * stride
                h_end = min(h_start + patch_size, padded_height)
                w_end = min(w_start + patch_size, padded_width)

                if h_end == padded_height:
                    h_start = padded_height - patch_size
                if w_end == padded_width:
                    w_start = padded_width - patch_size

                patch = image_pad[:, :, h_start:h_start + patch_size, w_start:w_start + patch_size]
                all_patches.append(patch)
                positions.append((h_start, w_start))

        mask_list = []
        with torch.no_grad():
            j = 1
            for i in range(0, len(all_patches), batch_patch_num):
                batch_patches = torch.cat(all_patches[i:i + batch_patch_num], dim=0)

                batch_resized = F.interpolate(batch_patches, (targetH, targetW), mode='bicubic', align_corners=False)

                mask_batch = generate_mask(batch_resized, model)

                mask_batch = F.interpolate(mask_batch, (patch_size, patch_size), mode='bilinear', align_corners=False)

                mask_list.append(mask_batch)

        all_masks = torch.cat(mask_list, dim=0) 

        cnt = 0
        for (h_start, w_start) in positions:
            mask_patch = all_masks[cnt:cnt + 1]
            cnt += 1
            mask_accum[:, :, h_start:h_start + patch_size, w_start:w_start + patch_size] += mask_patch * weight
            weight_accum[:, :, h_start:h_start + patch_size, w_start:w_start + patch_size] += weight

        mask_tensor = mask_accum / (weight_accum + 1e-8)
        mask_tensor = mask_tensor[:, :, :height, :width]

        return mask_tensor
    

    scale_masks = []
    for ps in patch_size_list:
        scale_mask = _seg_single_scale(ps)
        scale_masks.append(scale_mask)


    if len(scale_masks) == 1:
        mask_tensor = scale_masks[0]
    else:
        stack_masks = torch.stack(scale_masks, dim=0)

        if fusion_mode == 'mean':
            mask_tensor = (stack_masks.mean(dim=0)>= 0.5).float()

        elif fusion_mode == 'vote':
            binary = (stack_masks > vote_thresh).float()
            vote_score = binary.mean(dim=0)
            mask_tensor = (vote_score >= 0.5).float()

        elif fusion_mode == 'and':
            binary = (stack_masks > vote_thresh).float()
            mask_tensor = binary.min(dim=0).values

        else:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

    smoothing_kernel = torch.ones(1, 1, 3, 3, device=device) / 9.0
    mask_tensor = F.conv2d(mask_tensor, smoothing_kernel, padding=1)

    save_images(mask_tensor, './loc_results/{}_mask.png'.format(name)) 

    mask_tensor = mask_tensor.permute(0, 2, 3, 1)
    mask = mask_tensor.cpu().numpy()

    image_tensor, [rectified_height, rectified_width], angle = rectify(image, mask[0])

    return image_tensor, [rectified_height, rectified_width], angle, mask[0]

def save_images(images, path):
    images = images.cpu().data
    for i, image in enumerate(images):
        vutils.save_image(image, path)

def obtain_wm_blocks(image, seg_model, name, targetH=512, targetW=512):
    """
    return rectified watermarked blocks
    """
    image_tensor1, [height, width], angle, m = pad_split_seg_rectify_multi(image, seg_model, name, targetH=targetH, targetW=targetW)

    return image_tensor1.to(image.device)