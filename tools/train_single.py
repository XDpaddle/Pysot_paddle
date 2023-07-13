# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import time
import math
import json
import random
import numpy as np

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
import paddle
import paddle.nn as nn
from paddle.io import DataLoader 


from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler

from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.log_helper import init_log, print_speed, add_file_handler
from pysot.utils.distributed import dist_init, DistModule, reduce_gradients,\
        average_reduce, get_rank, get_world_size
from pysot.utils.model_load import load_pretrain, restore_from
from pysot.utils.average_meter import AverageMeter
from pysot.utils.misc import describe, commit
from pysot.models.model_builder import ModelBuilder
from pysot.datasets.dataset import TrkDataset
from pysot.core.config import cfg


logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--cfg', type=str, default='experiments\siamrpn_r50_l234_dwxcorr_8gpu\config.yaml',
                    help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')
args = parser.parse_args()

def build_data_loader():
    logger.info("build train dataset")
    # train_dataset
    train_dataset = TrkDataset()
    logger.info("build dataset done")

    train_sampler = None
    if get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,  # 改
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                            #   pin_memory=True,
                              sampler=train_sampler)
    return train_loader

def build_opt_lr(model, current_epoch=0):
    optim_params = model.parameters()
    lr_scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=cfg.TRAIN.BASE_LR, patience=3, verbose=True)  # 改
    # lr_scheduler = paddle.optimizer.lr.LinearWarmup(learing_rate, warmup_steps, start_lr, end_lr, last_epoch=- 1, verbose=False)
    optimizer = paddle.optimizer.SGD(learning_rate=lr_scheduler,parameters=optim_params)  # 未指定不同学习率
    return optimizer, lr_scheduler

def train(train_loader, model, optimizer, lr_scheduler):
    average_meter = AverageMeter()

    def is_valid_number(x):
        return not(math.isnan(x) or math.isinf(x) or x > 1e4)
    
    world_size = 1
    num_per_epoch = len(train_loader.dataset) // \
        cfg.TRAIN.EPOCH // (cfg.TRAIN.BATCH_SIZE * world_size)
    start_epoch = cfg.TRAIN.START_EPOCH
    epoch = start_epoch

    for idx, data in enumerate(train_loader):
        outputs = model(data)
        loss = outputs['total_loss']
        optimizer.step()
        optimizer.zero_grad()


def main():
    cfg.merge_from_file(args.cfg)
    model = ModelBuilder()
    model.train()
    train_loader = build_data_loader()
    lr_scheduler, optimizer = build_opt_lr(model)
    train(train_loader, optimizer, lr_scheduler)



    









if __name__ == '__main__':
    # seed_torch(args.seed)
    main()