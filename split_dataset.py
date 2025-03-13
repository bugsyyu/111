import os
import shutil
import random
from pathlib import Path
import logging

def setup_directories():
    """创建必要的目录结构"""
    directories = [
        'data/imgs',
        'data/masks',
        'data/test_images',
        'data/test_masks'
    ]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f'Created directory: {dir_path}')

def split_dataset(
    source_imgs_dir: str,
    source_masks_dir: str,
    train_ratio: float = 0.9,
    random_seed: int = 42
):
    """
    划分数据集为训练集和测试集
    
    Args:
        source_imgs_dir: 源图像目录
        source_masks_dir: 源掩码目录
        train_ratio: 训练集比例
        random_seed: 随机种子
    """
    # 设置随机种子
    random.seed(random_seed)
    
    # 获取所有图像文件
    image_files = list(Path(source_imgs_dir).glob('*.png'))
    if not image_files:
        image_files = list(Path(source_imgs_dir).glob('*.jpg'))
    
    if not image_files:
        raise RuntimeError(f'No image files found in {source_imgs_dir}')
    
    # 计算训练集大小
    total_size = len(image_files)
    train_size = int(total_size * train_ratio)
    
    # 随机打乱文件列表
    random.shuffle(image_files)
    
    # 划分训练集和测试集
    train_files = image_files[:train_size]
    test_files = image_files[train_size:]
    
    # 复制训练集文件
    for img_path in train_files:
        # 复制图像
        shutil.copy2(
            img_path,
            f'data/imgs/{img_path.name}'
        )
        # 复制对应的掩码
        mask_path = Path(source_masks_dir) / img_path.name.replace('.jpg', '_mask.png').replace('.png', '_mask.png')
        if mask_path.exists():
            shutil.copy2(
                mask_path,
                f'data/masks/{mask_path.name}'
            )
    
    # 复制测试集文件
    for img_path in test_files:
        # 复制图像
        shutil.copy2(
            img_path,
            f'data/test_images/{img_path.name}'
        )
        # 复制对应的掩码
        mask_path = Path(source_masks_dir) / img_path.name.replace('.jpg', '_mask.png').replace('.png', '_mask.png')
        if mask_path.exists():
            shutil.copy2(
                mask_path,
                f'data/test_masks/{mask_path.name}'
            )

    logging.info(f'Total images: {total_size}')
    logging.info(f'Training set: {len(train_files)} images')
    logging.info(f'Test set: {len(test_files)} images')

if __name__ == '__main__':
    import argparse
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Split dataset into train and test sets')
    parser.add_argument('--source-imgs', required=True, help='Source images directory')
    parser.add_argument('--source-masks', required=True, help='Source masks directory')
    parser.add_argument('--train-ratio', type=float, default=0.9, help='Training set ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # 创建目录结构
    setup_directories()
    
    # 执行数据集划分
    try:
        split_dataset(
            source_imgs_dir=args.source_imgs,
            source_masks_dir=args.source_masks,
            train_ratio=args.train_ratio,
            random_seed=args.seed
        )
    except Exception as e:
        logging.error(f'Error during dataset split: {str(e)}')
