import os.path
from setuptools import setup

HERE = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(HERE, "README.md")) as fid:
    README = fid.read()

setup(
    name="image_to_numpy",
    version="1.0.0",
    description="Load an image into a numpy array with proper Exif orientation handling",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/ageitgey/image_to_numpy",
    author="Adam Geitgey",
    author_email="ageitgey@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    packages=["image_to_numpy"],
    install_requires=[
        "pillow",
        "numpy"
    ],
)
