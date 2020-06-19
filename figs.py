import os
import glob
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
import torch

import models
import datasets
import decoder

def make_preds_labels(model_dir, save_file, utt_idx=0):
    '''
    Logs the difference between predicted and label word sequence of a single
    utterance in the dev set for every model saved in an experiment.
    '''
    with open(os.path.join(model_dir, 'args.json'), 'r') as f:
        args_dict = json.load(f)
    
    model = models.LSTM(num_layers=args_dict['n_layers'],
                        hidden_dim=args_dict['hidden_dim'],
                        bidirectional=(not args_dict['unidir']))

    wts = glob.glob(os.path.join(model_dir, '*.pt'))

    dataset = datasets.ESPnetBucketDataset(
                  os.path.join(args_dict['data_root'],
                               'dump/test_dev93/deltafalse/data.json'),
                  os.path.join(args_dict['data_root'],
                               'lang_1char/train_si284_units.txt'),
                  load_dir=args_dict['bucket_load_dir'],
                  n_buckets=args_dict['n_buckets'])    

    lines = {}
    for wt in wts:
        model.load_state_dict(torch.load(wt))
        device = torch.device('cpu')
        model.to(device)
        
        data = dataset[utt_idx]
        feat = data['feat'].copy()[None, ...]

        log_probs, embed = model(torch.tensor(feat))
        log_probs = log_probs.detach().numpy()
        labels = np.array(data['label'])

        preds, to_remove = decoder.batch_greedy_ctc_decode(log_probs,
                                                           zero_infinity=True)
        preds = preds[preds != to_remove]

        pred_words = decoder.compute_words(preds, dataset.idx2tok)
        label_words = decoder.compute_words(labels, dataset.idx2tok)

        lines[wt] = [] 
        lines[wt].append(f'Predicted:\n{" ".join(pred_words)}')
        lines[wt].append(f'Label:\n{" ".join(label_words)}')

    with open(os.path.join(model_dir, save_file), 'w') as f:
        for wt in lines.keys():
            f.write(f'{wt}\n')
            for line in lines[wt]:
                f.write(line + '\n')
            f.write('\n')

def make_epoch_plot(x, x_name, save_file):
    '''
    Plots a metric collected per epoch v.s. epoch.
    '''
    plt.plot(np.arange(len(x)), x)
    plt.title(f'{x_name} vs epoch')
    plt.savefig(save_file)
    plt.clf()

def check_valid(args):
    '''
    Checks that argument values are valid.
    '''
    assert os.path.isfile(os.path.join(args.model_dir, 'best.pt')), \
           f'best model best.pt must exist in args.model_dir: {args.model_dir}'
    assert (not os.path.exists(os.path.join(args.model_dir, args.save_file))), \
           f'args.save_file: {args.save_file} must be free'

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                 description='predictions for trained models across epoch')

    parser.add_argument('--model_dir',
                        type=str,
                        help='directory where models are saved')
    parser.add_argument('--save_file',
                        type=str,
                        help='file where predictions are saved',
                        default='model_preds.txt')

    args = parser.parse_args()

    check_valid(args)
    make_preds_labels(args.model_dir, args.save_file)
