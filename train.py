import os
import glob
import shutil
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from src.data import SynthTextDataset
from src.config import Config
from src.model import Model
from src.recorder import Recorder
from src.epoch import train_epoch

parser = argparse.ArgumentParser(description="Train a model with the train set of MJSynth")
parser.add_argument("data_path", help="path to the MJSynth dataset (directorry containing the file 'lexicon.txt')", type=str)
parser.add_argument("--restore", help="restore a model checkpoint", type=str, default=None)
args = parser.parse_args()


if not os.path.exists(os.path.join(Config.save_path, Config.session)):
    os.makedirs(os.path.join(Config.save_path, Config.session))
save_files = glob.glob(os.path.join("**", "*.py"), recursive=True)
path = os.path.join(Config.save_path, Config.session, Config.code_path)
os.makedirs(path, exist_ok=True)
for filepath in save_files:
    shutil.copy(filepath, path)

trainset = SynthTextDataset(args.data_path, "annotation_train.txt", "lexicon.txt")
trainset.max_seq_len = Config.max_seq_len
train_recorder = Recorder(os.path.join(Config.log_path, Config.session), "train")
trainloader = DataLoader(trainset, batch_size=Config.batch_size, shuffle=True, drop_last=True, num_workers=6)

model = Model()
model.to(Config.device)
if args.restore is not None:
    model.load_state_dict(torch.load(args.restore))
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
scheduler = ExponentialLR(optimizer, gamma=Config.learning_rate_decay)
loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")


for e in range(Config.epoch):
    train_epoch(trainloader, model, train_recorder, loss_fn, optimizer, scheduler)
