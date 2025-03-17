import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate_improved(net, dataloader, device, amp):
    """评估改进的U-Net模型在验证数据集上的表现"""
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # 在验证集上迭代
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='验证轮次', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # 将图像和标签移动到正确的设备和类型
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # 预测掩码
            if hasattr(net, 'training'):
                # 临时将网络设置为eval模式
                training_status = net.training
                net.eval()
                # 推理阶段无需计算误差图
                mask_pred = net(image)
                if isinstance(mask_pred, tuple):
                    mask_pred = mask_pred[0]  # 只取预测logits
                # 恢复原始训练状态
                net.train(training_status)
            else:
                mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, '真实掩码索引应在[0, 1]范围内'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # 计算Dice系数
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, '真实掩码索引应在[0, n_classes]范围内'
                # 转换为one-hot格式
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # 计算Dice系数，忽略背景
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    # 恢复训练模式
    if hasattr(net, 'training') and not net.training:
        net.train()

    return dice_score / max(num_val_batches, 1)