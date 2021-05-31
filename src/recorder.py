import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import resize

class Recorder:

    def __init__(self, tb_path, label):
        self.label = label
        self.running_loss = []
        self.running_ng = 0
        self.running_ok = 0
        self.running_loss_by_steps = {}
        self.running_ng_by_steps = {}
        self.running_ok_by_steps = {}
        self.writer = SummaryWriter(tb_path)

    def reset(self):
        self.running_loss = []
        self.running_ng = 0
        self.running_ok = 0
        self.running_loss_by_steps = {}
        self.running_ng_by_steps = {}
        self.running_ok_by_steps = {}

    def dump(self, epoch):
        loss = np.mean(self.running_loss)
        acc = self.running_ok / (self.running_ok + self.running_ng)
        print(f"subepoch {epoch} ({self.label}) - loss : {loss}, accuracy : {acc}")
        self.writer.add_scalar(f'{self.label}/loss/all', loss, epoch)
        self.writer.add_scalar(f'{self.label}/accuracy/all', acc, epoch)
        for key,value in self.running_loss_by_steps.items():
            self.writer.add_scalar(f'{self.label}/loss/{key}', np.mean(value), epoch)
        for key,ok in self.running_ok_by_steps.items():
            assert key in self.running_ng_by_steps
            ng = self.running_ng_by_steps[key]
            acc = ok / (ok + ng)
            self.writer.add_scalar(f'{self.label}/accuracy/{key}', acc, epoch)
        self.reset()
        return loss

    def record(self, step, loss, ok, ng):
        self.running_ok += ok
        self.running_ng += ng
        self.running_loss.append(loss)
        if step not in self.running_loss_by_steps:
            self.running_loss_by_steps[step] = []
        if step not in self.running_ng_by_steps:
            self.running_ng_by_steps[step] = 0
        if step not in self.running_ok_by_steps:
            self.running_ok_by_steps[step] = 0

        self.running_loss_by_steps[step].append(loss)
        self.running_ok_by_steps[step] += ok
        self.running_ng_by_steps[step] += ng

    def record_img(self, epoch, step, imgs, masks):
        self.writer.add_images(
            f'{self.label}/images/{step}',
            imgs,
            epoch)
        masks = masks.permute([0,3,1,2])
        masks = resize(masks, imgs.shape[2:])


        grayscales = imgs.mean(dim=1).unsqueeze(dim=1)
        tgs = grayscales * (1 - masks)
        masked_imgs = torch.cat([masks.pow(1/4), tgs*0.5 + masks.sqrt()*0.5, tgs*0.5], dim=1)
        self.writer.add_images(
            f'{self.label}/masks/{step}',
            masked_imgs,
            epoch)

