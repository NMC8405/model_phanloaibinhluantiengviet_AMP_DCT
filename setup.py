# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name="phobert_cnn_metadata_sentiment",
    version="1.0.0",
    description="Phân loại cảm xúc bình luận tiếng Việt - PhoBERT + CNN + Metadata",
    packages=find_packages(),
    python_requires=">=3.10",
    py_modules=["config"],
)
