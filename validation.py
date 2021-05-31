import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import SynthTextDataset
from src.config import Config
from src.model import Model
from src.recorder import Recorder
from src.epoch import val_epoch


parser = argparse.ArgumentParser(description="Evaluate a model on the validation set of MJSynth")
parser.add_argument("data_path", help="path to the MJSynth dataset (directorry containing the file 'lexicon.txt')", type=str)
parser.add_argument("model", help="path to the trained model checkpoint to load", type=str)
args = parser.parse_args()

valset = SynthTextDataset("/hdd/OCR/MJSynth/data", "annotation_val.txt", "lexicon.txt")
valset.max_seq_len = Config.max_seq_len
val_recorder = Recorder(os.path.join(Config.log_path, Config.session), "validation")
valloader = DataLoader(valset, batch_size=Config.batch_size, shuffle=True, drop_last=True, num_workers=6)

model = Model()
model.to(Config.device)
model.load_state_dict(torch.load(args.model, map_location=Config.device))
model.eval()
loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

with torch.no_grad():
    val_epoch(valloader, model, val_recorder, loss_fn)
