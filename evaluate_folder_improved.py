import argparse
import logging
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

from predict_improved import predict_img, mask_to_image
from improved_unet import ImprovedUNet


def plot_img_and_mask_to_file(img, mask, output_path):
    """将图像和掩码可视化并保存到文件"""
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1, figsize=(12, 4))
    ax[0].set_title('输入图像')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'掩码 (类别 {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.savefig(output_path)
    plt.close()


def evaluate_folder(
        model_path: str,
        input_folder: str,
        mask_folder: str,
        output_folder: str,
        device: str = 'cuda',
        viz: bool = False,
        no_save: bool = False,
        mask_threshold: float = 0.5,
        scale: float = 0.5,
        n_classes: int = 2,
        bilinear: bool = False
):
    """评估模型在文件夹中所有图像上的性能"""
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    viz_folder = os.path.join(output_folder, 'visualizations')
    if viz:
        os.makedirs(viz_folder, exist_ok=True)

    # 加载模型
    net = ImprovedUNet(n_channels=3, n_classes=n_classes, bilinear=bilinear)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    net.to(device=device)

    logging.info(f'加载模型 {model_path}')
    logging.info(f'使用设备 {device}')

    state_dict = torch.load(model_path, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    logging.info('模型已加载!')

    # 获取所有图片文件
    input_files = list(Path(input_folder).glob('*.png')) + list(Path(input_folder).glob('*.jpg'))
    if not input_files:
        raise RuntimeError(f'在 {input_folder} 中未找到输入文件')

    # 初始化评估指标
    all_pa = []  # 像素准确率
    all_iou = []  # 交并比
    all_true = []
    all_pred = []

    # 批量处理
    for input_path in tqdm(input_files, desc='处理图像'):
        filename = str(input_path)
        logging.info(f'\n预测图像 {filename} ...')

        # 加载并预测图片
        img = Image.open(filename)
        mask = predict_img(
            net=net,
            full_img=img,
            scale_factor=scale,
            out_threshold=mask_threshold,
            device=device
        )

        # 保存预测结果
        if not no_save:
            out_filename = str(Path(output_folder) / input_path.name)
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'掩码已保存到 {out_filename}')

        # 保存可视化结果
        if viz:
            viz_filename = os.path.join(viz_folder, f'{input_path.stem}_visualization.png')
            plot_img_and_mask_to_file(img, mask, viz_filename)
            logging.info(f'可视化已保存到 {viz_filename}')

        # 计算评估指标
        mask_path = Path(mask_folder) / input_path.name.replace('.jpg', '_mask.png').replace('.png', '_mask.png')
        if mask_path.exists():
            true_mask = np.array(Image.open(mask_path))
            if true_mask.max() > 1:
                true_mask = true_mask > 127

            # 计算PA和IoU
            pa, iou = calculate_metrics(mask, true_mask)
            all_pa.append(pa)
            all_iou.append(iou)
            all_true.append(true_mask.flatten())
            all_pred.append(mask.flatten())

    # 计算并保存评估结果
    if all_pa:
        save_evaluation_results(all_pa, all_iou, all_true, all_pred, output_folder)


def calculate_metrics(pred_mask, true_mask):
    """计算像素准确率和交并比"""
    # 转换为二值
    pred_mask = pred_mask > 0
    true_mask = true_mask > 0

    # 计算IoU
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    iou = intersection / union if union > 0 else 0

    # 计算PA
    total_pixels = pred_mask.size
    correct_pixels = (pred_mask == true_mask).sum()
    pa = correct_pixels / total_pixels

    return pa, iou


def save_evaluation_results(all_pa, all_iou, all_true, all_pred, output_folder):
    """保存评估结果"""
    # 计算平均指标
    mean_pa = np.mean(all_pa)
    mean_iou = np.mean(all_iou)

    # 计算整体指标
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    # 保存混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'))
    plt.close()

    # 保存评估结果
    results = {
        '平均像素准确率': mean_pa,
        '平均IoU': mean_iou,
        '精确率': precision,
        '召回率': recall,
        '混淆矩阵': cm.tolist()
    }

    with open(os.path.join(output_folder, 'evaluation_results.txt'), 'w', encoding='utf-8') as f:
        f.write('评估结果\n')
        f.write('-' * 50 + '\n')
        for metric, value in results.items():
            if metric != '混淆矩阵':
                f.write(f'{metric}: {value:.4f}\n')
        f.write('\n混淆矩阵:\n')
        f.write(str(cm))

    # 保存Excel文件兼容格式的结果
    try:
        import pandas as pd
        df = pd.DataFrame({
            '指标': ['平均像素准确率', '平均IoU', '精确率', '召回率'],
            '值': [mean_pa, mean_iou, precision, recall]
        })
        df.to_csv(os.path.join(output_folder, 'evaluation_results.csv'), index=False, encoding='utf-8-sig')
    except ImportError:
        logging.warning('未安装pandas库，跳过保存CSV结果')


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='在图像文件夹上评估改进的U-Net模型')
    parser.add_argument('--model', '-m', default='MODEL.pth', help='模型路径')
    parser.add_argument('--input', '-i', required=True, help='输入文件夹')
    parser.add_argument('--masks', '-ma', required=True, help='真实掩码文件夹')
    parser.add_argument('--output', '-o', required=True, help='输出文件夹')
    parser.add_argument('--viz', '-v', action='store_true', help='可视化预测')
    parser.add_argument('--no-save', '-n', action='store_true', help='不保存预测')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5, help='掩码阈值')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='缩放因子')
    parser.add_argument('--device', '-d', default='cuda', help='使用的设备')
    parser.add_argument('--classes', '-c', type=int, default=2, help='类别数')
    parser.add_argument('--bilinear', '-b', action='store_true', default=False, help='使用双线性上采样')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    try:
        evaluate_folder(
            model_path=args.model,
            input_folder=args.input,
            mask_folder=args.masks,
            output_folder=args.output,
            device=args.device,
            viz=args.viz,
            no_save=args.no_save,
            mask_threshold=args.mask_threshold,
            scale=args.scale,
            n_classes=args.classes,
            bilinear=args.bilinear
        )
    except Exception as e:
        logging.error(f'评估失败: {str(e)}')