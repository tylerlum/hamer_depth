from pathlib import Path

from setuptools import find_packages, setup

VERSION = "0.1.0"
DESCRIPTION = "HaMeR Depth: Accurate Hand Pose Estimation with HaMeR + Refinement with Depth Images"
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="hamer_depth",
    version=VERSION,
    author="Tyler Lum",
    author_email="tylergwlum@gmail.com",
    url="https://github.com/tylerlum/hamer_depth",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[],
    keywords=[
        "hand pose estimation",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
)
