""" 改进的U-Net模型 """

import torch
import torch.nn as nn
import torch.nn.functional as F
from .improved_unet_parts import *


class ImprovedUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(ImprovedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 编码器
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # 解码器
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # 动态特征融合模块（应用在最后一层解码器特征上）
        self.dffm = DynamicFeatureFusionModule(64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, mask=None):
        # 编码器路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码器路径
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # 应用动态特征融合模块
        enhanced_features, error_map = self.dffm(x)

        # 生成最终输出
        logits = self.outc(enhanced_features)

        # 如果是训练阶段，返回额外信息用于损失函数计算
        if self.training and mask is not None:
            # 计算真实误差图
            with torch.no_grad():
                pred_probs = F.softmax(logits, dim=1) if self.n_classes > 1 else torch.sigmoid(logits)

                if self.n_classes > 1:
                    pred_mask = pred_probs[:, 1:].sum(dim=1, keepdim=True)
                    true_mask = F.one_hot(mask, self.n_classes)[:, :, :, 1:].sum(dim=3, keepdim=True).permute(0, 3, 1,
                                                                                                              2)
                else:
                    pred_mask = pred_probs
                    true_mask = mask.unsqueeze(1).float()

                # 阈值化预测概率
                pred_binary = (pred_mask >= 0.5).float()

                # 计算真实误差图 E=(Y-Ŷ)·(Y-Ŷ')
                error_true = (true_mask - pred_mask) * (true_mask - pred_binary)

            return logits, error_map, error_true

        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.dffm = torch.utils.checkpoint(self.dffm)
        self.outc = torch.utils.checkpoint(self.outc)