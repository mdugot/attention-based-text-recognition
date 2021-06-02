import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from src.data import SynthTextDataset
from src.config import Config
from src.model import Model
from src.recorder import Recorder
from src.epoch import val_epoch


parser = argparse.ArgumentParser(description="Evaluate a model on the validation set of MJSynth")
parser.add_argument("data_path", help="path to the MJSynth dataset (directorry containing the file 'lexicon.txt')", type=str)
parser.add_argument("model", help="path to the trained model checkpoint to load", type=str)
parser.add_argument("--plot", help="plot the attention maske for each step of the inference", action="store_true")
args = parser.parse_args()

if args.plot:
    Config.batch_size = 1
    Config.cycle = 1

valset = SynthTextDataset("/hdd/OCR/MJSynth/data", "annotation_val.txt", "lexicon.txt")
valset.max_seq_len = Config.max_seq_len
val_recorder = Recorder(os.path.join(Config.log_path, Config.session), "validation", plot=args.plot)
valloader = DataLoader(valset, batch_size=Config.batch_size, shuffle=True, drop_last=True)

model = Model()
model.to(Config.device)
model.load_state_dict(torch.load(args.model, map_location=Config.device))
model.eval()
loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

with torch.no_grad():
    try:
        val_epoch(valloader, model, val_recorder, loss_fn, args.plot)
    except KeyboardInterrupt:
        if args.plot:
            plt.ioff()
            plt.close('all')
