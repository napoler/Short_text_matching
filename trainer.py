# -*- coding: utf-8 -*-
"""
作者：　terrychan
Blog: https://terrychan.org
# 说明：

v2
训练核心入口


"""
import os
from pytorch_lightning.utilities.cli import LightningCLI
# from model.myModel import ShortMatch
from model.rdropModel import ShortMatch

# os.environ['TOKENIZERS_PARALLELISM'] = "true"


if __name__ == '__main__':
    # freeze_support()
    # https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html

    cli = LightningCLI(ShortMatch, save_config_overwrite=True)