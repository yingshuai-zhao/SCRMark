import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips


class WatermarkLoss(nn.Module):
    def __init__(self, device, lambda1, lambda2, lambda3):
        super(WatermarkLoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.loss_lpips = lpips.LPIPS(net='vgg').eval().to(device)

    def forward(self, Pw, Ir, Iw, imgs, imgs_w):
        # 计算Pattern损失
        Lzero = self.near_zero_loss(Pw)
        Llpips = self.loss_lpips(imgs_w, imgs.detach()).mean()
        
        # 重新组合Pattern损失
        Lpattern = (self.lambda1 * Lzero )

        # 计算Message损失
        Lmessage = self.message_loss(Ir, Iw)

        # 总损失
        total_loss = self.beta * Lpattern + self.gamma * Lmessage

        return total_loss, Lpattern, Lmessage, Llpips

    def near_zero_loss(self, Pw):
        # 保持原有的近零损失
        target = torch.zeros_like(Pw)
        Lzero = F.mse_loss(Pw, target)
        return Lzero

    def message_loss(self, Ir, Iw):
        # 保持原有的信息损失
        Lmessage = nn.BCEWithLogitsLoss()(Ir, Iw)
        return Lmessage


class WatermarkDecodingConsistencyLoss(nn.Module):
    """
    水印解码导向的特征一致性损失
    重点保证最终解码层特征的稳定性
    """
    
    def __init__(self, temperature=0.1):
        super(WatermarkDecodingConsistencyLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, noisy_features, clean_features):
        """
        Args:
            clean_features: 各层解码特征 [[b, c, h, w], ...]，最后一个是解码输出
            noisy_features: 对应的噪声特征
            watermark_bits: 真实水印位 [b, n_bits]，用于重要性加权
        """
        total_loss = 0.0
        num_levels = len(clean_features)
        
        for _, (clean_feat, noisy_feat) in enumerate(zip(clean_features, noisy_features)):

            loss = self.feature_consistency(clean_feat, noisy_feat)
            
            total_loss += loss
        
        return total_loss / num_levels
    
    
    def feature_consistency(self, clean_feat, noisy_feat):
        """中间特征层的一致性"""
        batch_size, channels, height, width = clean_feat.shape
        
        # 通道维度归一化，保持空间结构
        clean_norm = F.normalize(clean_feat, p=2, dim=1)
        noisy_norm = F.normalize(noisy_feat, p=2, dim=1)
        
        # 空间一致性
        # spatial_similarity = torch.sum(clean_norm * noisy_norm, dim=1)
        spatial_loss = F.mse_loss(noisy_norm, clean_norm.detach())
        
        return spatial_loss


 
bce_loss = torch.nn.BCEWithLogitsLoss(reduce=False)
def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v) * (labels_v + 1)
    loss1 = bce_loss(d1, labels_v) * (labels_v + 1)
    loss2 = bce_loss(d2, labels_v) * (labels_v + 1)
    loss3 = bce_loss(d3, labels_v) * (labels_v + 1)
    loss4 = bce_loss(d4, labels_v) * (labels_v + 1)
    loss5 = bce_loss(d5, labels_v) * (labels_v + 1)
    loss6 = bce_loss(d6, labels_v) * (labels_v + 1)
    loss = torch.mean(3*loss0 + 0.25 * (loss1 + loss2 + loss3 + loss4 + loss5 + loss6))
    return loss


def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()


def muti_iou_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    iou0 = iou_loss(d0, labels_v)
    iou1 = iou_loss(d1, labels_v)
    iou2 = iou_loss(d2, labels_v)
    iou3 = iou_loss(d3, labels_v)
    iou4 = iou_loss(d4, labels_v)
    iou5 = iou_loss(d5, labels_v)
    iou6 = iou_loss(d6, labels_v)
    loss = torch.mean(3*iou0 + 0.25*(iou1 + iou2 + iou3 + iou4 + iou5 + iou6))
    return loss