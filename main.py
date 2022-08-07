# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import datetime
import json
import math
import os
import random
import sys
import time
from pathlib import Path
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as nnf
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
from models import ViT as vit
# import vision_transformer as vits

try:
    import wandb
except ImportError:
    wandb = None


torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))


def get_args_parser():
    parser = argparse.ArgumentParser('Your args here', add_help=False)

    # must have this arg for run_with_waic
    parser.add_argument('--output_dir', default=None, type=str, help='Path to save logs and checkpoints.')

    # need this one for multi GPU
    parser.add_argument("--dist_url", default="env://",  # setup using environment variables
                        type=str, help='url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html')

    return parser


def main_function_on_each_process(local_rank, args):
    print(f'running main function multiple times {local_rank}: {args}\n\n')
    os.environ['RANK'] = f'{local_rank}'
    os.environ['LOCAL_RANK'] = f'{local_rank}'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '56782'
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)

    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(path_to_data, transform=transform)

    # Must - for data parallel
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,  # when True first iter at job ~250 sec, then ~30 sec. other iters faster.
        drop_last=False,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building  networks ... ============
    # define your model here
    model = MyNetwork(...)

    # move networks to gpu
    model = model.to(device=f'cuda:{local_rank}')

    # This is the sync part - must have for multi GPU
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])


    # ============ preparing optimizer ... ============
    optimizer = torch.optim.SGD(model.params, lr=learning_rate, momentum=0.9)

    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model=model,
        optimizer=optimizer
    )
    start_epoch = to_restore["epoch"]

    # ============ w and b logger ============
    # logging/saving checkpoints etc are only on main process!
    if utils.is_main_process() and wandb is not None:
        try:
            wandb.init(project=args.arch, id=os.path.basename(args.output_dir),
                       resume=os.path.basename(args.output_dir))
        except Exception as e:
            print(f'could not start wandb. got {e}')
            sys.exit(1)

    start_time = time.time()
    print("Starting !")
    for epoch in range(start_epoch, args.epochs):
        # MUST for multi GPU to set epoch for the distributed sampler
        data_loader.sampler.set_epoch(epoch)
        # ============ training one epoch  ... ============
        train_stats = train_one_epoch(model,
            data_loader, optimizer,
            epoch, args)
        # ============ writing logs ... ============
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args
        }
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {} rank={}'.format(total_time_str, args.rank), flush=True)
    # do not forget to shut down wandb
    if utils.is_main_process() and wandb is not None:
        wandb.finish()


def train_one_epoch(model, data_loader, ):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, targets) in enumerate(metric_logger.log_every(data_loader, 50, header)):

        # move images to gpu
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        pred = model(images)
        loss = loss_fn(pred, targets)

        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        if utils.is_main_process() and wandb is not None:
            wandb.log({'loss': loss.item(), 'lr': optimizer.param_groups[0]["lr"],
                       'wd': optimizer.param_groups[0]["weight_decay"]})
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger, flush=True)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if wandb is not None:
        # see: https://github.com/wandb/client/blob/master/docs/dev/wandb-service-user.md
        wandb.require("service")
        wandb.setup()

    # do not spawn if world size is 1 or less
    if torch.cuda.device_count() <= 1:
        main_function_on_each_process(torch.cuda.current_device(), args=args)
    else:
        # ready to spawn
        os.environ['WORLD_SIZE'] = f'{torch.cuda.device_count()}'
        torch.multiprocessing.spawn(main_function_on_each_process, nprocs=torch.cuda.device_count(), args=(args,))
