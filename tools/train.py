# Copyright (c) general_backbone. All rights reserved.
#!/usr/bin/env python
import argparse
import os
import os.path as osp

import torch
import datetime
from general_backbone import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import yaml
import logging

import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torch.utils.data.dataloader import DataLoader
import torchvision
from general_backbone.data import create_dataset, create_loader, Mixup, FastCollateMixup, AugMixDataset
from general_backbone.models import create_model, safe_model_name 
from general_backbone.utils import resume_checkpoint, load_checkpoint, model_parameters
from general_backbone.layers import convert_splitbn_model
from general_backbone.utils import *
from general_backbone.loss import *
from general_backbone.optim import create_optimizer_v2, optimizer_kwargs
from general_backbone.scheduler import create_scheduler
from general_backbone.utils import ApexScaler, NativeScaler
from general_backbone.data.loader import create_loader_aug
from general_backbone.utils import ConfigDict, Config
from general_backbone.data import AlbImageDataset, AugmentationDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError: 
    has_wandb = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')


parser = argparse.ArgumentParser(description='General backbone model training', add_help=False)
parser.add_argument('-c', '--config', default=None, type=str, metavar='FILE',
                    help='python config file specifying default arguments')

# General Config
parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet50")')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')

# Checkpoint
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')

# Logging
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')
parser.add_argument("--local-rank", default=0, type=int)


# DataLoader & Dataset
parser.add_argument('--data-dir', type=str, default='toydata/image_classification',
                    help='Link to root directory dataset')
parser.add_argument('--img-size', type=int, default=224,
                    help='Input image size')
parser.add_argument('-b', '--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('-j', '--num-workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--pin-memory', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--shuffle', action='store_true', default=False,
                    help='Is shuffle dataset before training')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                    help='amount to decay each learning rate cycle (default: 0.5)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit, cycles enabled if > 1')
parser.add_argument('--lr-k-decay', type=float, default=1.0,
                    help='learning rate k-decay for cosine/poly (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# ---------------------------------------------------

# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=2e-5,
                    help='weight decay (default: 2e-5)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = parser.parse_known_args()

    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = Config(filename=args_config.config)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_config, args_text

# setup_default_logging()
args, args_config, args_text = _parse_args()

# Update args into config
if args_config.config is None:
    cfg = Config.from_argparser(args)
else:
    cfg = Config.fromfile(args_config.config)

print(cfg)
cfg_train = cfg.train_conf
cfg_test = cfg.test_conf
data_config_train = cfg.data_conf.data.train
data_config_test = cfg.data_conf.data.eval

def main():
    
    # initialize model
    model = create_model(
        cfg_train.model,
        pretrained=cfg_train.pretrained,
        num_classes=cfg_train.num_classes,
        checkpoint_path=cfg_train.initial_checkpoint
        )

    # move model to GPU
    model.cuda()

    if cfg_train.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if cfg_train.local_rank == 0:
        print(f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')    

    dataset_train = AugmentationDataset(data_dir=cfg.data_root,
                            name_split=data_config_train.name_split,
                            config_file = args.config, 
                            dict_transform=cfg.data_conf.dict_transform, 
                            input_size=(cfg.data_conf.img_size, cfg.data_conf.img_size),
                            debug=data_config_train.debug,
                            dir_debug=data_config_train.dir_debug, 
                            class_2_idx=cfg.data_conf.class_2_idx)

    dataset_eval = AugmentationDataset(data_dir=cfg.data_root,
                            name_split=data_config_test.name_split,
                            config_file = args.config, 
                            dict_transform=cfg.data_conf.dict_transform, 
                            input_size=(cfg.data_conf.img_size, cfg.data_conf.img_size),
                            class_2_idx=cfg.data_conf.class_2_idx)

    # Dataloader
    loader_train = DataLoader(
        dataset_train,
        batch_size=cfg_train.batch_size,
        shuffle=cfg_train.shuffle,
        num_workers=cfg_train.num_workers,
        pin_memory=cfg_train.pin_memory,
        prefetch_factor=cfg_train.prefetch_factor)

    
    loader_eval = DataLoader(
        dataset_eval,
        batch_size=cfg_test.batch_size,
        shuffle=cfg_test.shuffle,
        num_workers=cfg_test.num_workers,
        pin_memory=cfg_test.pin_memory,
        prefetch_factor=cfg_test.prefetch_factor)

    # Optimizer and Scheduler
    optimizer = torch.optim.Adam(
        [{
            'params': model.parameters()
        }],
        lr=cfg_train.lr,
        weight_decay=1e-6)

    # setup loss function
    
    train_loss_fn = nn.CrossEntropyLoss().cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # optionally resume from a checkpoint
    resume_epoch = None
    if cfg_train.resume:
        resume_epoch = resume_checkpoint(
            model, cfg_train.resume,
            optimizer=None if cfg_train.no_resume_opt else optimizer,
            log_info=cfg_train.local_rank == 0)

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(cfg_train, optimizer)
    start_epoch = 0
    if cfg_train.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = cfg_train.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    # setup checkpoint saver and eval metric tracking
    eval_metric = cfg_train.eval_metric
    best_metric = None
    best_epoch = None
    output_dir = None
    
    exp_name = '-'.join([
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        safe_model_name(cfg_train.model),
        str(cfg.data_conf.img_size)
    ])
    
    output_dir = get_outdir(cfg_train.output if cfg_train.output else './output/train', exp_name)
    decreasing = True if eval_metric == 'loss' else False
    
    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)

    try:
        for epoch in range(start_epoch, num_epochs):

            train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, cfg_train,
                lr_scheduler=lr_scheduler, output_dir=output_dir)

            eval_metrics = validate(model, loader_eval, validate_loss_fn, cfg_train)
            
            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if output_dir is not None:
                update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None, log_wandb=cfg_train.log_wandb and has_wandb)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, output_dir=None, amp_autocast=suppress, model_ema=None, mixup_fn=None):

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        
        input, target = input.cuda(), target.cuda()
        if mixup_fn is not None:
            input, target = mixup_fn(input, target)
    
        with amp_autocast():
            output = model(input)
            loss = loss_fn(output, target)

        losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
    
        loss.backward(create_graph=second_order)
        optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % cfg_train.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if cfg_train.local_rank == 0:
                print('Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) / batch_time_m.val,
                        rate_avg=input.size(0) / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])



def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            input = input.cuda()
            target = target.cuda()

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            
            reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if cfg_train.local_rank == 0 and (last_batch or batch_idx % cfg_train.log_interval == 0):
                log_name = 'Test' + log_suffix
                print(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics

if __name__ == '__main__':
    main()
