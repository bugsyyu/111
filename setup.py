from setuptools import setup, find_packages

setup(
    name="improved-unet-segmentation",
    version="1.0.0",
    author="论文作者",
    author_email="author@example.com",
    description="基于误差预测和动态特征融合的改进U-Net光学薄膜损伤检测",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/author/Improved-UNet",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "numpy>=1.23.5",
        "Pillow>=9.3.0",
        "tqdm>=4.64.1",
        "wandb>=0.13.5",
        "matplotlib>=3.6.2",
        "scikit-learn>=1.0.0",
        "seaborn>=0.12.0",
    ],
)