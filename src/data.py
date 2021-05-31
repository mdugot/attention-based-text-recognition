import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, Compose, CenterCrop
from torchvision.transforms.functional import pad
import pandas
from PIL import Image
from tqdm import tqdm

from .config import Config


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return pad(image, padding, 0, 'constant')


class SynthTextDataset(Dataset):
    def __init__(self, root, annotation, lexicon):
        self.chars = "abcdefghijklmnopqrstuvwxyz0123456789"
        self.max_seq_len = 0
        self.transform = Compose([
            SquarePad(),
            Resize(Config.resize_shape),
            CenterCrop(Config.img_shape),
            ToTensor()
        ])
        annotation_df = pandas.read_csv(os.path.join(root, annotation), names=["filename", "lexic"], delimiter=" ")
        lexicon_df = pandas.read_csv(os.path.join(root, lexicon), names=["lexic"], dtype={"lexic":str}, keep_default_na=False)
        self.files = []
        self.labels = []
        for idx in tqdm(range(len(annotation_df))):
            filename = os.path.join(root, annotation_df["filename"][idx])
            lexic_idx = annotation_df["lexic"][idx]
            text = lexicon_df["lexic"][lexic_idx]
            label = [self.chars.index(c) for c in text]
            if len(label) > self.max_seq_len:
                self.max_seq_len = len(label)
            self.files.append(filename)
            self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try :
            img = Image.open(self.files[idx])
            tensor = self.transform(img)
            img.close()
            label = self.labels[idx][:self.max_seq_len]
            while len(label) < self.max_seq_len:
                label.append(Config.nchars - 1)
            return {"image": tensor, "label": torch.tensor(label)}
        except:
            return self[np.random.randint(len(self.files))]
