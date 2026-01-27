"""Setup script for zunda package."""

from setuptools import setup, find_packages

setup(
    name="zunda",
    version="0.1.0",
    description="東北ずん子project画像データセット用ライブラリ",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "torchvision",
        "pillow",
    ],
)
