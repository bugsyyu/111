import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from improved_unet import ImprovedUNet, ImprovedUNetLoss
from utils.data_loading import BasicDataset, CarvanaDataset

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        edge_weight: float = 0.5,
        aux_weight: float = 0.3,
        num_workers: int = 2,  # 减少工作线程数量
):
    # 1. 创建数据集
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. 划分训练集和验证集
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. 创建数据加载器，限制工作线程数量
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 初始化wandb日志 - 可以在这里手动捕获wandb异常
    try:
        experiment = wandb.init(project='ImprovedU-Net', resume='allow', anonymous='must')
        experiment.config.update(
            dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                 val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                 amp=amp, edge_weight=edge_weight, aux_weight=aux_weight)
        )
    except Exception as e:
        logging.warning(f"wandb初始化失败: {e}. 继续训练但不记录日志。")
        experiment = None

    logging.info(f'''开始训练:
        轮次:          {epochs}
        批大小:      {batch_size}
        学习率:   {learning_rate}
        训练集大小:   {n_train}
        验证集大小: {n_val}
        保存检查点:     {save_checkpoint}
        设备:          {device.type}
        图像缩放:  {img_scale}
        混合精度: {amp}
        边缘损失权重: {edge_weight}
        辅助损失权重: {aux_weight}
        工作线程数: {num_workers}
    ''')

    # 4. 设置优化器、损失函数、学习率调度器和AMP的损失缩放
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # 目标：最大化Dice得分
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # 使用改进的损失函数
    criterion = ImprovedUNetLoss(n_classes=model.n_classes, edge_weight=edge_weight, aux_weight=aux_weight)
    global_step = 0

    # 5. 开始训练
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'轮次 {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'网络定义了 {model.n_channels} 个输入通道, ' \
                    f'但加载的图像有 {images.shape[1]} 个通道。请检查图像是否正确加载。'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                # 清除缓存以释放内存
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    # 前向传播（在训练模式下，模型会返回额外的误差信息）
                    try:
                        outputs = model(images, true_masks)
                        if isinstance(outputs, tuple) and len(outputs) == 3:
                            pred_logits, error_pred, error_true = outputs
                            # 计算总损失（包括交叉熵、边缘损失和辅助损失）
                            loss, loss_dict = criterion(pred_logits, true_masks, error_pred, error_true)
                        else:
                            # 兼容旧版本模型
                            pred_logits = outputs
                            if model.n_classes == 1:
                                loss = F.binary_cross_entropy_with_logits(pred_logits, true_masks.float())
                            else:
                                loss = F.cross_entropy(pred_logits, true_masks)
                            loss_dict = {'total_loss': loss.item()}
                    except RuntimeError as e:
                        # 处理内存不足错误，跳过这个批次
                        if "out of memory" in str(e):
                            logging.error(f"CUDA内存不足: {e}")
                            if device.type == 'cuda':
                                torch.cuda.empty_cache()
                            continue
                        else:
                            raise e

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()

                # 防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

                grad_scaler.unscale_(optimizer)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss_dict['total_loss']

                # 记录各部分损失
                if experiment is not None:
                    try:
                        log_dict = {f'train_{k}': v for k, v in loss_dict.items()}
                        log_dict.update({'step': global_step, 'epoch': epoch})
                        experiment.log(log_dict)
                    except Exception as e:
                        logging.warning(f"wandb日志记录失败: {e}")

                pbar.set_postfix(**{'loss (batch)': loss_dict['total_loss']})

                # 验证轮次 - 减少验证频率以节省内存
                division_step = max(n_train // (2 * batch_size), 1)
                if global_step % division_step == 0:
                    # 清理内存
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

                    # 计算验证得分
                    val_score = evaluate(model, val_loader, device, amp)
                    scheduler.step(val_score)
                    logging.info('验证Dice得分: {}'.format(val_score))

                    # 尝试记录到wandb，但如果失败则继续训练
                    if experiment is not None:
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'step': global_step,
                                'epoch': epoch,
                            })

                            # 只在每个epoch末记录图像以节省内存
                            if global_step % (n_train // batch_size) == 0:
                                experiment.log({
                                    'images': wandb.Image(images[0].cpu()),
                                    'masks': {
                                        'true': wandb.Image(true_masks[0].float().cpu()),
                                        'pred': wandb.Image(
                                            pred_logits.argmax(dim=1)[0].float().cpu() if model.n_classes > 1
                                            else (torch.sigmoid(pred_logits[0]) > 0.5).float().cpu()),
                                    }
                                })
                        except Exception as e:
                            logging.warning(f"wandb记录验证数据失败: {e}")

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            # 减少检查点大小，只保存必要的数据
            state_dict = {k: v.cpu() for k, v in state_dict.items()}
            state_dict['mask_values'] = dataset.mask_values

            # 避免每轮都保存，只保存第1轮、最后一轮和每5轮的检查点
            if epoch == 1 or epoch == epochs or epoch % 5 == 0:
                torch.save(state_dict, str(dir_checkpoint / f'improved_unet_checkpoint_epoch{epoch}.pth'))
                logging.info(f'检查点 {epoch} 已保存!')


def get_args():
    parser = argparse.ArgumentParser(description='使用改进的U-Net模型在图像和掩码上进行训练')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='训练轮次')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='批大小')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='学习率', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='从.pth文件加载模型')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='图像的缩放因子')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='用作验证的数据百分比 (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='使用混合精度')
    parser.add_argument('--bilinear', action='store_true', default=False, help='使用双线性上采样')
    parser.add_argument('--classes', '-c', type=int, default=2, help='类别数')
    parser.add_argument('--edge-weight', '-ew', type=float, default=0.5, help='边缘损失的权重')
    parser.add_argument('--aux-weight', '-aw', type=float, default=0.3, help='辅助损失的权重')
    parser.add_argument('--num-workers', '-nw', type=int, default=2, help='数据加载器工作线程数')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'使用设备 {device}')

    # 设置较小的内存分数，避免内存溢出
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.7)  # 限制GPU内存使用，留出30%给系统

    # 根据参数创建改进的U-Net模型
    model = ImprovedUNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    # 使用通道优先格式以优化性能
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'网络:\n'
                 f'\t{model.n_channels} 输入通道\n'
                 f'\t{model.n_classes} 输出通道（类别）\n'
                 f'\t{"双线性" if model.bilinear else "转置卷积"} 上采样')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        if 'mask_values' in state_dict:
            del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'模型已从 {args.load} 加载')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            edge_weight=args.edge_weight,
            aux_weight=args.aux_weight,
            num_workers=args.num_workers
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('检测到内存不足错误! '
                      '请降低批处理大小或图像尺寸。尝试使用以下命令重新训练:\n'
                      'python train_improved.py --batch-size 4 --amp')

        # 清理缓存
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # 启用检查点以减少内存使用
        model.use_checkpointing()
        try:
            # 以更小的批处理大小重新训练
            batch_size = max(args.batch_size // 4, 1)
            logging.info(f'正在使用较小的批处理大小 ({batch_size}) 和检查点重新尝试...')
            train_model(
                model=model,
                epochs=args.epochs,
                batch_size=batch_size,  # 显著减小批处理大小
                learning_rate=args.lr,
                device=device,
                img_scale=args.scale,
                val_percent=args.val / 100,
                amp=True,  # 强制启用AMP
                edge_weight=args.edge_weight,
                aux_weight=args.aux_weight,
                num_workers=1  # 减少工作线程
            )
        except Exception as e:
            logging.error(f'第二次尝试也失败: {e}\n'
                          f'请考虑手动降低批处理大小和图像比例，并确保您的系统有足够的虚拟内存。')
    except Exception as e:
        logging.error(f'训练过程中发生错误: {e}')