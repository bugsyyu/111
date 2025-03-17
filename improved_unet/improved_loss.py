""" 改进的U-Net损失函数 """

import torch
import torch.nn as nn
import torch.nn.functional as F
from .improved_unet_parts import calculate_edge


class ImprovedUNetLoss(nn.Module):
    """
    改进的U-Net损失函数，包含像素级交叉熵损失、边缘损失和辅助损失

    Args:
        n_classes (int): 类别数
        edge_weight (float): 边缘损失的权重系数
        aux_weight (float): 辅助损失的权重系数
    """

    def __init__(self, n_classes=2, edge_weight=0.5, aux_weight=0.3):
        super(ImprovedUNetLoss, self).__init__()
        self.n_classes = n_classes
        self.edge_weight = edge_weight
        self.aux_weight = aux_weight

        # 像素级交叉熵损失或二元交叉熵损失
        self.ce_loss = nn.CrossEntropyLoss() if n_classes > 1 else nn.BCEWithLogitsLoss()

        # 误差预测子网络的辅助损失（MSE）
        self.aux_loss = nn.MSELoss()

    def forward(self, pred_logits, true_mask, error_pred, error_true):
        """
        计算总损失

        Args:
            pred_logits (Tensor): 模型预测的logits，形状为[B, C, H, W]
            true_mask (Tensor): 真实掩码，形状为[B, H, W]（对于多类别）或[B, 1, H, W]（对于二值）
            error_pred (Tensor): 预测的误差图，形状为[B, 1, H, W]
            error_true (Tensor): 真实的误差图，形状为[B, 1, H, W]

        Returns:
            loss (Tensor): 总损失
            loss_dict (dict): 包含各组成部分损失的字典
        """
        # 计算像素级交叉熵损失
        if self.n_classes > 1:
            ce_loss = self.ce_loss(pred_logits, true_mask)
        else:
            ce_loss = self.ce_loss(pred_logits, true_mask.float())

        # 计算边缘损失
        if self.n_classes > 1:
            # 对于多类别分割，使用argmax获取预测掩码
            pred_mask = torch.argmax(pred_logits, dim=1, keepdim=True)
            pred_edge = calculate_edge(pred_mask.float())

            # 将真实掩码转换为单通道
            if true_mask.dim() == 3:
                true_mask_edge = calculate_edge(true_mask.unsqueeze(1).float())
            else:
                true_mask_edge = calculate_edge(true_mask.float())
        else:
            # 对于二值分割，使用sigmoid将logits转换为概率
            pred_mask = torch.sigmoid(pred_logits)
            pred_edge = calculate_edge(pred_mask)
            true_mask_edge = calculate_edge(true_mask.float())

        # 使用误差图作为权重来计算边缘损失
        # 在误差大的区域给予更高的权重
        edge_loss = torch.mean(error_true * torch.pow(pred_edge - true_mask_edge, 2))

        # 计算辅助损失（误差预测的MSE）
        auxiliary_loss = self.aux_loss(error_pred, error_true)

        # 计算总损失
        total_loss = ce_loss + self.edge_weight * edge_loss + self.aux_weight * auxiliary_loss

        # 返回总损失和各部分损失（用于记录）
        loss_dict = {
            'ce_loss': ce_loss.item(),
            'edge_loss': edge_loss.item(),
            'aux_loss': auxiliary_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict