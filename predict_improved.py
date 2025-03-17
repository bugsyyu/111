import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from improved_unet import ImprovedUNet
from utils.utils import plot_img_and_mask


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    """预测单张图像的掩码"""
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        if isinstance(output, tuple):
            output = output[0]  # 只取预测logits

        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用改进的U-Net从输入图像预测掩码')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='指定存储模型的文件')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='输入图像的文件名', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='输出图像的文件名')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='可视化处理的图像')
    parser.add_argument('--no-save', '-n', action='store_true', help='不保存输出掩码')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='将掩码像素视为白色的最小概率值')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='输入图像的缩放因子')
    parser.add_argument('--bilinear', action='store_true', default=False, help='使用双线性上采样')
    parser.add_argument('--classes', '-c', type=int, default=2, help='类别数')

    return parser.parse_args()


def get_output_filenames(args):
    """获取输出文件名"""

    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    """将掩码数组转换为图像"""
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)

    net = ImprovedUNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'加载模型 {args.model}')
    logging.info(f'使用设备 {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('模型已加载!')

    for i, filename in enumerate(in_files):
        logging.info(f'预测图像 {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'掩码已保存到 {out_filename}')

        if args.viz:
            logging.info(f'可视化图像 {filename} 的结果, 关闭窗口继续...')
            plot_img_and_mask(img, mask)