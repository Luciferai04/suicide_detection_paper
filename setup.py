from pathlib import Path

from setuptools import setup, find_packages

setup(
    name="suicide-detection-research",
    version="0.1.0",
    description="Research code for suicide risk detection with SVM, BiLSTM, and BERT.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[],
)

