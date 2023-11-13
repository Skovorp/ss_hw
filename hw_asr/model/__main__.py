from hw_asr.model.my_spex_model import MySpExModel
import argparse
import collections
import warnings

import numpy as np
import torch
from torch import nn

import hw_asr.loss as module_loss
import hw_asr.metric as module_metric
import hw_asr.model as module_arch
from hw_asr.trainer import Trainer
from hw_asr.utils import prepare_device
from hw_asr.utils.object_loading import get_dataloaders
from hw_asr.utils.parse_config import ConfigParser


config = ConfigParser.from_path('/home/ubuntu/ss_hw/hw_asr/configs/one_batch_test.json')
logger = config.get_logger("train")

# setup data_loader instances
dataloaders = get_dataloaders(config)
n_train_speakers = len(dataloaders['train'].dataset.all_speakers)
device = torch.device("cuda")
model = config.init_obj(config["arch"], module_arch, n_train_speakers=n_train_speakers).to(device)


def move_batch_to_device(batch, device: torch.device):
    """
    Move all necessary tensors to the HPU
    """
    batch['audios']['mix'] = batch['audios']['mix'].to(device)
    batch['audios']['refs'] = batch['audios']['refs'].to(device)
    batch['audios']['targets'] = batch['audios']['targets'].to(device)
    batch['speaker_ids'] = batch['speaker_ids'].to(device)

    return batch

print("starting loop")
for i, b in enumerate(dataloaders['val']):
    if i <= 18:
        continue
    print(i)
    b = move_batch_to_device(b, device)
    res = model(**b, predict_speaker=False)