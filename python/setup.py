#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="visualgeometry",
    version="0.1.0",
    description="Python interface for VisualGeometryCore.jl - circles and conics with NumPy",
    author="VisualGeometryCore Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "juliacall>=0.9.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)