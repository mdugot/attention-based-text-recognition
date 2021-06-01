import os
import torch
from tqdm import tqdm

from src.config import Config


def _epoch(dataloader, model, recorder, loss_fn, training, optimizer=None, scheduler=None, plot=False):
    if training:
        assert optimizer is not None
        assert scheduler is not None
    pbar = tqdm(total=Config.cycle)
    nbatch = 0
    subepoch = 0
    for b_idx, batch in enumerate(dataloader):
        states, previous_chars = model.init_state()
        imgs = batch["image"].to(Config.device)
        labels = batch["label"].to(Config.device)
        if training:
            optimizer.zero_grad()
        for step in range(Config.max_seq_len):
            not_end = labels[:,step] != (Config.nchars - 1)
            imgs = imgs[not_end]
            if len(imgs) ==  0:
                break
            labels = labels[not_end]
            states = (states[0][:,not_end], states[1][:,not_end])
            previous_chars = previous_chars[not_end]

            outputs, states, masks = model(imgs, states, previous_chars)
            if nbatch == 0:
                recorder.record_img(subepoch, step, imgs, masks)
                if plot:
                    recorder.plot(step, imgs, masks, outputs.argmax(dim=1))
            loss = loss_fn(outputs, labels[:,step]) / Config.batch_size
            ok = torch.sum(outputs.argmax(dim=1) == labels[:,step]).item()
            ng =  len(labels) - ok
            recorder.record(step, loss.item(), ok, ng)
            if training:
                loss.backward(retain_graph=True)
            previous_chars = labels[:,step]
        if training:
            optimizer.step()
        nbatch += 1
        if nbatch >= Config.cycle:
            pbar.update(1)
            pbar.close()
            recorder.dump(subepoch)
            pbar = tqdm(total=Config.cycle)
            nbatch = 0
            subepoch += 1
            if training:
                scheduler.step()
                torch.save(
                    model.state_dict(),
                    os.path.join(Config.save_path, Config.session, f'{subepoch}.chkpt')
                )
        else :
            pbar.update(1)


def train_epoch(dataloader, model, recorder, loss_fn, optimizer, scheduler):
    _epoch(dataloader, model, recorder, loss_fn, True, optimizer, scheduler)

def val_epoch(dataloader, model, recorder, loss_fn, plot):
    _epoch(dataloader, model, recorder, loss_fn, False, plot=plot)
