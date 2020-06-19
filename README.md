## CTC-based ASR on the )Wall Street Journal dataset

This code trains an LSTM with connectionist temporal classification (CTC) to do
character-based automatic speech recognition (ASR) on the Wall Street Journal
(WSJ).

### Environment

This code is tested in
```
python 3.8.1
```
with the additional packages:
```
numpy 1.18.1
torch 1.5.0a0+857bae3
kaldiio 2.15.1
matplotlib 3.1.3
tqdm 4.43.0
```

### Running the code

In addition to these dependencies, in order to run the code, you will need
access to the WSJ dataset (both [WSJ0](https://catalog.ldc.upenn.edu/LDC93S6A) 
and [WSJ1](https://catalog.ldc.upenn.edu/LDC94S13A)), preprocessed by 
[ESPnet](https://github.com/espnet/espnet) (see the script 
`espnet/egs/wsj/asr1/run.sh` in that repository).

#### Training:

You can train a model with:
```
python run.py --data_root=path/to/dataset --temp_root=path/to/temporarily/copy/data/to --model_dir=path/to/save/models/and/logs 
``` 
which will train and evaluate on the train and development set.


#### Evaluating

You can evaluate a trained model with:
```
python run.py --model_dir /path/to/saved/models/and/logs --eval_only
```
which will evaluate on the train and development set.

#### Additional info

- Note, when training, the code copies the data to a local directory for faster
  reads. To clean up the temporary data when training finishes,
  add the flag `--cleanup`.
- To also evaluate on the test set, add the flag `--test`.
- To run only on the cpu, add the flag `--cpu`.
- For a full list of arguments, try `python run.py --help`.

### Files
- `run.py`: the main training and evaluation routines for the ASR model on WSJ
  dataset
- `datasets.py`: PyTorch Dataset, and Sampler related code
- `models.py`: a PyTorch LSTM model
- `decoder.py`: code for taking the output probabilities of an acoustic model,
  decoding it to characters and words, and evaluating character error rate and
  word error rate
- `transforms.py`: code for preprocessing the data on the fly before training
- `utils.py`: utility functions for saving, copying, debugging and more
- `figs.py`: code for generating plots and logs of model predictions over time
