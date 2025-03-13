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

from predict import predict_img, mask_to_image
from unet import UNet

def plot_img_and_mask_to_file(img, mask, output_path):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1, figsize=(12, 4))
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
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
    scale: float = 0.5
):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    viz_folder = os.path.join(output_folder, 'visualizations')
    if viz:
        os.makedirs(viz_folder, exist_ok=True)
    
    # 加载模型
    net = UNet(n_channels=3, n_classes=2)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    
    logging.info(f'Loading model {model_path}')
    logging.info(f'Using device {device}')
    
    state_dict = torch.load(model_path, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    logging.info('Model loaded!')
    
    # 获取所有图片文件
    input_files = list(Path(input_folder).glob('*.png')) + list(Path(input_folder).glob('*.jpg'))
    if not input_files:
        raise RuntimeError(f'No input file found in {input_folder}')
    
    # 初始化评估指标
    all_pa = []
    all_iou = []
    all_true = []
    all_pred = []
    
    # 批量处理
    for input_path in tqdm(input_files, desc='Processing images'):
        filename = str(input_path)
        logging.info(f'\nPredicting image {filename} ...')
        
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
            logging.info(f'Mask saved to {out_filename}')
        
        # 保存可视化结果
        if viz:
            viz_filename = os.path.join(viz_folder, f'{input_path.stem}_visualization.png')
            plot_img_and_mask_to_file(img, mask, viz_filename)
            logging.info(f'Visualization saved to {viz_filename}')
        
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
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'))
    plt.close()
    
    # 保存评估结果
    results = {
        'Mean Pixel Accuracy': mean_pa,
        'Mean IoU': mean_iou,
        'Precision': precision,
        'Recall': recall,
        'Confusion Matrix': cm.tolist()
    }
    
    with open(os.path.join(output_folder, 'evaluation_results.txt'), 'w') as f:
        f.write('Evaluation Results\n')
        f.write('-' * 50 + '\n')
        for metric, value in results.items():
            if metric != 'Confusion Matrix':
                f.write(f'{metric}: {value:.4f}\n')
        f.write('\nConfusion Matrix:\n')
        f.write(str(cm))

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate model on a folder of images')
    parser.add_argument('--model', '-m', default='MODEL.pth', help='Path to model')
    parser.add_argument('--input', '-i', required=True, help='Input folder')
    parser.add_argument('--masks', '-ma', required=True, help='Ground truth masks folder')
    parser.add_argument('--output', '-o', required=True, help='Output folder')
    parser.add_argument('--viz', '-v', action='store_true', help='Visualize predictions')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save predictions')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5, help='Mask threshold')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Scale factor')
    parser.add_argument('--device', '-d', default='cuda', help='Device to use')
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
            scale=args.scale
        )
    except Exception as e:
        logging.error(f'Evaluation failed: {str(e)}') 