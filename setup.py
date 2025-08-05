#!/usr/bin/env python3
"""Setup script for DarkOperator Studio."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="darkoperator",
    version="0.1.0",
    author="Daniel Schmidt",
    author_email="daniel@terragonlabs.com",
    description="Neural Operators for Ultra-Rare Dark Matter Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danieleschmidt/darkoperator-studio",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.12.0",
        "scipy>=1.8.0",
        "matplotlib>=3.5.0",
        "pandas>=1.4.0",
        "scikit-learn>=1.1.0",
        "h5py>=3.6.0",
        "tqdm>=4.64.0",
        "pyyaml>=6.0",
        "requests>=2.28.0",
        "uproot>=4.3.0",
        "awkward>=1.8.0",
        "plotly>=5.10.0",
        "dash>=2.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
        ],
        "physics": [
            "pyroot>=6.24.0",
            "particle>=0.20.0",
            "hepstats>=0.6.0",
            "pylhe>=0.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "darkoperator=darkoperator.cli:main",
        ],
    },
)