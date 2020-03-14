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

import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet.contrib import amp
import pdb
from mxnet import gluon, nd, init, context
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from mxboard import SummaryWriter
from mxnet.contrib import amp

from gluoncv.utils import makedirs, LRScheduler, LRSequential, split_and_load
from gluoncv.data.tracking_data.track import TrkDataset
from gluoncv.model_zoo import get_model
from gluoncv.loss import SiamRPNLoss
from gluoncv.utils.parallel import *
from mxnet.contrib import amp
os.system("export MXNET_CUDNN_AUTOTUNE_DEFAULT=0")

def parse_args():
    """parameter test."""
    parser = argparse.ArgumentParser(description='siamrpn tracking test result')
    parser.add_argument('--model_name', type=str, default='siamrpn_alexnet_v2_otb15',
                        help='name of model.')
    parser.add_argument('--use-pretrained', action='store_true',default=False,
                        help='enable using pretrained model from gluon.')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    parser.add_argument('--save-dir', type=str, default='params',
                        help='directory of saved models')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--ngpus', type=int, default=2,
                        help='number of gpus to use.')
    parser.add_argument('--resume-params', type=str, default=None,
                        help='path of parameters to load from.')
    parser.add_argument('--logging-file', type=str, default='train.log',
                        help='name of training log file')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=32, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs.')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='training start epochs.')   
    parser.add_argument('--lr-mode', type=str, default='step',
                        help='lr mode')
    parser.add_argument('--base-lr', type=float, default=0.005,
                        help='base lr')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='base lr')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 1e-4)') 
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, \
                        and beta/gamma for batchnorm layers.')
    parser.add_argument('--cls-weight', type=float, default=1.0,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--loc-weight', type=float, default=1.2,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--amp', action='store_true',
                        help='Use MXNet AMP for mixed precision training.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--syncbn', action='store_true', default=False,
                        help='using Synchronized Cross-GPU BatchNorm')
    parser.add_argument('--kvstore', type=str, default='device',
                        help='kvstore to use for trainer/module.')
    parser.add_argument('--accumulate', type=int, default=1,
                        help='new step to accumulate gradient. If >1, the batch size is enlarged.')
    parser.add_argument('--use-amp', action='store_true',
                        help='whether to use automatic mixed precision.')
    parser.add_argument('--no-val', action='store_true', default=True,
                            help='skip validation during training')
    parser.add_argument('--mode', type=str, default='hybrid',
                    help='mode in which to train the model. options are symbolic, imperative, hybrid')
    parser.add_argument('--is-train', type=str, default=True,
                    help='if train the model. options are True, False')
    opt = parser.parse_args()

    if opt.no_cuda:
        print('Using CPU')
        opt.kvstore = 'local'
        opt.ctx = [mx.cpu(0)]
    else:
        print('Number of GPUs:', opt.ngpus)
        assert opt.ngpus > 0, 'No GPUs found, please enable --no-cuda for CPU mode.'
        opt.ctx = [mx.gpu(i) for i in range(opt.ngpus)]

    # logging and checkpoint saving   
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    # Synchronized BatchNorm
    opt.norm_layer = mx.gluon.contrib.nn.SyncBatchNorm if opt.syncbn \
        else mx.gluon.nn.BatchNorm
    opt.norm_kwargs = {'num_devices': opt.ngpus} if opt.syncbn else {}
    return opt

# opt = parse_args()
# logger = logging.getLogger('global')
def train_batch_fn(data, opt):
    template = split_and_load(data[0], ctx_list=opt.ctx, batch_axis=0)
    search = split_and_load(data[1], ctx_list=opt.ctx, batch_axis=0)
    label_cls = split_and_load(data[2], ctx_list=opt.ctx, batch_axis=0)
    label_loc = split_and_load(data[3], ctx_list=opt.ctx, batch_axis=0)
    label_loc_weight = split_and_load(data[4], ctx_list=opt.ctx, batch_axis=0)
    return template, search, label_cls, label_loc, label_loc_weight
    

def build_data_loader(batch_size):
    # dataset and dataloader
    logger.info("build train dataset")
    # train_dataset
    train_dataset = TrkDataset()
    logger.info("build dataset done")

    train_sampler = None
    train_loader = gluon.data.DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=opt.num_workers)
    print(len(train_loader))
    # pdb.set_trace()
    return train_loader

def main(logger, opt):
    filehandler = logging.FileHandler(os.path.join(opt.save_dir, opt.logging_file))
    streamhandler = logging.StreamHandler()
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    logger.info(opt)

    sw = SummaryWriter(logdir=opt.save_dir, flush_secs=5, verbose=False)

    if opt.kvstore is not None:
        kv = mx.kvstore.create(opt.kvstore)
        logger.info('Distributed training with %d workers and current rank is %d' % (kv.num_workers, kv.rank))
    if opt.use_amp:
        amp.init()

    num_gpus = opt.ngpus
    batch_size = opt.batch_size*max(1,num_gpus)
    logger.info('Total batch size is set to %d on %d GPUs' % (batch_size, num_gpus))
    # num_gpus = opt.num_gpus
    # batch_size *= max(1, num_gpus)
    # logger.info('Total batch size is set to %d on %d GPUs' % (batch_size, num_gpus))
    # context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]

    train_loader = build_data_loader(batch_size)

    # create model
    net = get_model(opt.model_name, bz=opt.batch_size, if_train=opt.is_train, ctx=opt.ctx)
    net.cast(opt.dtype)
    logger.info(net)
    if opt.resume_params is not None:
        if os.path.isfile(opt.resume_params):
            net.load_parameters(opt.resume_params, ctx=context)
            print('Continue training from model %s.' % (opt.resume_params))
        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(opt.resume))

    # create criterion
    criterion = SiamRPNLoss(opt.batch_size, opt.ctx)
    
    step_epoch = [10,20,30,40,50]
    num_batches = len(train_loader)
    lr_scheduler = LRSequential([LRScheduler(mode='step',
                                            base_lr=0.005,
                                            target_lr=0.01,
                                            nepochs=opt.warmup_epochs,
                                            iters_per_epoch=num_batches,
                                            step_epoch = step_epoch,
                                            ),
                                LRScheduler(mode='poly',
                                            base_lr=0.01,
                                            target_lr=0.005,
                                            nepochs=opt.epochs-opt.warmup_epochs,
                                            iters_per_epoch=num_batches,
                                            step_epoch = [e - opt.warmup_epochs for e in step_epoch],
                                            power=0.02)])

    kv = mx.kv.create(opt.kvstore)
    optimizer_params = {'lr_scheduler': lr_scheduler,
                        'wd': opt.weight_decay,
                        'momentum': opt.momentum,
                        'learning_rate': opt.lr}

    if opt.dtype == 'float32':
        optimizer_params['multi_precision'] = True

    if opt.use_amp:
        amp.init_trainer(optimizer_params)

    if opt.no_wd:
        for k, v in net.module.collect_params('.*beta|.*gamma|.*bias').items():
            v.wd_mult = 0.0
    
    # if opt.mode == 'hybrid':
    #     net.hybridize(static_alloc=True, static_shape=True)

    optimizer = gluon.Trainer(net.collect_params(), 'sgd', optimizer_params, kvstore=kv)

    if opt.accumulate > 1:
        params = [p for p in net.collect_params().values() if p.grad_req != 'null']
        for p in params:
            p.grad_req = 'add'

    train(opt, net, train_loader, criterion, optimizer, batch_size, kv, logger)

def save_checkpoint(net, opt, epoch, is_best=False):
    """Save Checkpoint"""
    filename = 'epoch_%d.params'%(epoch)
    filepath = os.path.join(opt.save_dir, filename)
    net.save_parameters(filepath)
    if is_best:
        shutil.copyfile(filename, os.path.join(opt.save_dir, 'model_best.params'))   

def train(opt, net, train_loader, criterion, trainer, batch_size, kv, logger):
    for epoch in range(opt.start_epoch, opt.epochs):
        loss_total_val= 0
        loss_loc_val= 0
        loss_cls_val= 0
        batch_time = time.time()
        for i, data in enumerate(train_loader):
            template, search, label_cls, label_loc, label_loc_weight=train_batch_fn(data, opt)
            cls_losses = []
            loc_losses = []
            total_losses = []
            with autograd.record():
                for j in range(len(opt.ctx)):
                    cls, loc=net(template[j], search[j])
                    cls_loss, loc_loss = criterion(cls, loc, label_cls[j], label_loc[j], label_loc_weight[j])
                    total_loss = opt.cls_weight*cls_loss+opt.loc_weight*loc_loss
                    cls_losses.append(cls_loss)
                    loc_losses.append(loc_loss)
                    total_losses.append(total_loss)
                mx.nd.waitall()
                if opt.amp:
                    with amp.scale_loss(total_losses, trainer) as scaled_loss:
                        autograd.backward(scaled_loss)
                else:
                    autograd.backward(total_losses)
            if opt.accumulate > 1 and (i + 1) % opt.accumulate == 0:
                if opt.kvstore is not None:
                    trainer.step(batch_size * kv.num_workers * opt.accumulate)
                else:
                    trainer.step(batch_size * opt.accumulate)
                    net.collect_params().zero_grad()
            else:
                if opt.kvstore is not None:
                    trainer.step(batch_size * kv.num_workers)
                else:
                    trainer.step(batch_size)
            trainer.step(opt.batch_size)
            loss_total_val += sum([l.mean().asscalar() for l in total_losses]) / len(total_losses)
            loss_loc_val += sum([l.mean().asscalar() for l in loc_losses]) / len(loc_losses)
            loss_cls_val += sum([l.mean().asscalar() for l in cls_losses]) / len(cls_losses)
            if i%2==0:
                logger.info('Epoch %d iteration %04d/%04d: loc loss %.3f, cls loss %.3f, training loss %.3f, batch time %.3f'% \
                                (epoch, i, len(train_loader), loss_loc_val/(i+1), loss_cls_val/(i+1),
                                loss_total_val/(i+1), time.time()-batch_time))
                batch_time = time.time()
                mx.nd.waitall()
        if opt.no_val:
            save_checkpoint(net, opt, epoch, False)

if __name__ == '__main__':
    logger = logging.getLogger('global')
    opt = parse_args()
    main(logger, opt)
