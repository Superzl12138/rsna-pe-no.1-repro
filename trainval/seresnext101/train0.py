import argparse
import numpy as np
import pandas as pd
import os
import sys
import cv2
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from apex import amp
from pretrainedmodels.senet import se_resnext101_32x4d
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import random
import pickle
import albumentations
import copy
from pathlib import Path
from transformers import get_linear_schedule_with_warmup

sys.path.append(str(Path(__file__).resolve().parents[1]))
from seresnext_input_utils import build_image_triplet, get_frangi_config, print_frangi_config

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class PEDataset(Dataset):
    def __init__(self, image_dict, bbox_dict, image_list, target_size, transform, frangi_config):
        self.image_dict=image_dict
        self.bbox_dict=bbox_dict
        self.image_list=image_list
        self.target_size=target_size
        self.transform=transform
        self.frangi_config=frangi_config
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self,index):
        x = build_image_triplet(
            image_dict=self.image_dict,
            bbox_dict=self.bbox_dict,
            center_image_id=self.image_list[index],
            target_size=self.target_size,
            frangi_config=self.frangi_config,
        )
        x = self.transform(image=x)['image']
        x = x.transpose(2, 0, 1)
        y = self.image_dict[self.image_list[index]]['pe_present_on_image']
        return x, y

class seresnext101(nn.Module):
    def __init__(self ):
        super().__init__()
        self.net = se_resnext101_32x4d(num_classes=1000, pretrained='imagenet')
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        in_features = self.net.last_linear.in_features
        self.last_linear = nn.Linear(in_features, 1)
    def forward(self, x):
        x = self.net.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", "--local-rank", dest="local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    args = parser.parse_args()
    if args.local_rank == -1:
        args.local_rank = int(os.environ.get("LOCAL_RANK", -1))
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.device = device

    seed = 2001
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # prepare input
    import pickle
    with open('../process_input/split2/image_list_train.pickle', 'rb') as f:
        image_list_train = pickle.load(f) 
    with open('../process_input/split2/image_dict.pickle', 'rb') as f:
        image_dict = pickle.load(f) 
    with open('../lung_localization/split2/bbox_dict_train.pickle', 'rb') as f:
        bbox_dict_train = pickle.load(f) 
    print(len(image_list_train), len(image_dict), len(bbox_dict_train))

    # hyperparameters
    learning_rate = 0.0004
    batch_size = int(os.environ.get('TRAIN_BATCH_SIZE_SERESNEXT101', 12))
    image_size = 576
    num_epoch = 1
    frangi_config = get_frangi_config()
    print_frangi_config(frangi_config)

    # build model
    if args.local_rank != 0:
        torch.distributed.barrier()
    model = seresnext101()
    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    num_train_steps = int(len(image_list_train)/(batch_size*4)*num_epoch)   # 4 GPUs
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    criterion = nn.BCEWithLogitsLoss().to(args.device)

    # training
    train_transform = albumentations.Compose([
        albumentations.RandomContrast(limit=0.2, p=1.0),
        albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, p=1.0),
        albumentations.Cutout(num_holes=2, max_h_size=int(0.4*image_size), max_w_size=int(0.4*image_size), fill_value=0, always_apply=True, p=1.0),
        albumentations.Normalize(mean=(0.456, 0.456, 0.456), std=(0.224, 0.224, 0.224), max_pixel_value=255.0, p=1.0)
    ])

    # iterator for training
    datagen = PEDataset(image_dict=image_dict, bbox_dict=bbox_dict_train, image_list=image_list_train, target_size=image_size, transform=train_transform, frangi_config=frangi_config)
    sampler = DistributedSampler(datagen)
    generator = DataLoader(dataset=datagen, sampler=sampler, batch_size=batch_size, num_workers=5, pin_memory=True)

    for ep in range(num_epoch):
        losses = AverageMeter()
        model.train()
        for j,(images,labels) in enumerate(generator):
            images = images.to(args.device)
            labels = labels.float().to(args.device)

            logits = model(images)
            loss = criterion(logits.view(-1),labels)
            losses.update(loss.item(), images.size(0))

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            scheduler.step()

        if args.local_rank == 0:
            print('epoch: {}, train_loss: {}'.format(ep,losses.avg), flush=True)

        if args.local_rank == 0:
            out_dir = 'weights/'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            torch.save(model.module.state_dict(), out_dir+'epoch{}'.format(ep))

if __name__ == "__main__":
    main()
