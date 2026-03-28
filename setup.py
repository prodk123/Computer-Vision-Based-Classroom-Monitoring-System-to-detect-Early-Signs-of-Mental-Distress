"""Setup script for Classroom Monitoring System."""

from setuptools import setup, find_packages

setup(
    name="classroom-monitor",
    version="1.0.0",
    description=(
        "Computer Vision-Based Classroom Monitoring System for Early Detection "
        "of Distress and Concentration Risk Indicators in Students"
    ),
    author="Research Team",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "Pillow>=10.0.0",
        "PyYAML>=6.0",
        "omegaconf>=2.3.0",
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.14.0",
    ],
    entry_points={
        "console_scripts": [
            "cm-train=scripts.train:main",
            "cm-evaluate=scripts.evaluate:main",
            "cm-preprocess=scripts.preprocess:main",
            "cm-infer=scripts.run_inference:main",
        ],
    },
)
