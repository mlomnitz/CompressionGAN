#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


NAME = 'CompressionGAN'
DESCRIPTION = 'Image compression using GANs',
MAINTAINER = 'Michael R Lomnitz'
MAINTAINER_EMAIL = 'mllomnitz@gmail.com'
URL = 'https://github.com/mlomnitz/CompressionGAN'
LICENSE = 'MIT'

setup(name=NAME,
      version='0.0.1',
      description=DESCRIPTION,
      license=LICENSE,
#      long_description=LONG_DESCRIPTION,
      author=MAINTAINER,
      author_email=MAINTAINER_EMAIL,
      url=URL,
      packages=find_packages(),
      scripts=[
        'scripts/train.py',
        'scripts/compress.py'
        ]
    )
