# Combination of CNN and LSTM for Attention-based text recognition

Implementation in Pytorch of an attention-based combination of CNN and LSTM to read a text character by character.

<p align="center">
  <img src="https://github.com/mdugot/attention-based-text-recognition/blob/master/plot.gif" />
</p>

## Dataset

Download the [MJSynth dataset](https://www.robots.ox.ac.uk/~vgg/data/text/) to train or evaluate the model.

## Training

Use the command `python3 train.py <path_to_the_dataset>` to run a training on the train set of the MJSynth dataset.
During the training, the model state will be regularly saved in the folder named `./saves` inside the subfolder corresponding to the date the training started.
A training can be continued from a previous state with the option `--restore <path_to_model_checkpoint>`
More info can be dislayed with `python3 train.py -h`.

```
>>> python3 train.py -h
usage: train.py [-h] [--restore RESTORE] data_path

Train a model with the train set of MJSynth

positional arguments:
  data_path          path to the MJSynth dataset (directorry containing the file 'lexicon.txt')

optional arguments:
  -h, --help         show this help message and exit
  --restore RESTORE  restore a model checkpoint
```
## Validation

Use the command `python3 validation.py <path_to_the_dataset> <path_to_model_checkpoint>` to evaluate a trained model on the validation set of the MJSynth dataset.
Use the option `--plot` to plot an animation of the attention mask at each step of the inference.
The file `trained.chkpt` contains the state of a trained model that can directly be used for the validation.
More info can be dislayed with `python3 validation.py -h`.

```
>>> python3 validation.py -h
usage: validation.py [-h] [--plot] data_path model

Evaluate a model on the validation set of MJSynth

positional arguments:
  data_path   path to the MJSynth dataset (directorry containing the file 'lexicon.txt')
  model       path to the trained model checkpoint to load

optional arguments:
  -h, --help  show this help message and exit
  --plot      plot the attention maske for each step of the inference
```
