# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 10:23:23 2020

@author: Lukas Baischer
"""

from setuptools import setup, find_packages

setup(name='intuitus_converter',
      version='0.1',
      author="Lukas Baischer",
      author_email="lukas_baischer@gmx.at",
      license="GPLv3",
      description="""Model converter for Intuitus: An FPGA based CNN hardware accelerator""",
      url='https://github.com/LukiBa/Intuitus-converter.git',
      long_description_content_type="text/markdown",
      packages=find_packages(),
      package_data={
          # If any package contains *.txt or *.rst files, include them:
          '': ['*.txt', '*.rst', '*.i', '*.c', '*.h', '*.json', '*.tcl', '*.vhd'],
      },
      test_requires=['numpy', 'wget', 'pathlib', 'json', 'bitstring', 'setuptools', 'argparse', 'typing', 'scikit-learn', 'terminaltables',\
                    'idx2numpy', 'pandas', 'brewer2mpl', 'tensorflow>=2.1', 'torch>=1.8', 'torchsummary', 'torchvision', 'matplotlib', 'tqdm', 'opencv-python>=4.4', 'easydict'],
      install_requires=['setuptools', 'numpy', 'wget', 'pathlib', 'bitstring', 'argparse', 'typing', 'scikit-learn', 'terminaltables',\
                        'matplotlib', 'pandas', 'brewer2mpl','idx2numpy', 'tensorflow>=2.1', 'torch>=1.8', 'torchsummary', 'torchvision', 'tqdm', 'opencv-python>=4.4', 'easydict'])