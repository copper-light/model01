## setup.py
from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

setup(
    name='cnn_model',
    version='0.1',
    packages=find_packages()
)

