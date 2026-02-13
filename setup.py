"""Setup script for VLA encoder ablation study."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text()
else:
    long_description = "VLA Vision Encoder Ablation Study"

setup(
    name="vla-encoder",
    version="0.1.0",
    description="Vision encoder ablation study for vision-language-action policies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="VLA Research Team",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pillow>=9.5.0",
        "transformers>=4.30.0",
        "sentencepiece>=0.1.99",
        "timm>=0.9.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vla-train=scripts.train:main",
            "vla-eval=scripts.evaluate:main",
            "vla-ablation=scripts.run_ablation:main",
            "vla-generate-data=scripts.generate_synthetic_data:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
