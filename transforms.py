import os
import json

import numpy as np
import tqdm

import utils
import datasets

class Normalize(object):
    '''
    Normalize each utterance by its own mean and standard deviation.
    '''
    def __init__(self):
        pass
    
    def __call__(self, sample):
        feat = sample['feat']
        sample['feat'] = ((feat - np.mean(feat, axis=0)) 
                           / np.std(feat, axis=0)).astype(np.float32)
        return sample

class SpeakerNormalize(object):
    '''
    Normalize each utterance by the mean and standard deviation of its speaker.
    '''
    def __init__(self, spk2meanstd):
        self.spk2meanstd = spk2meanstd

    def __call__(self, sample):
        mean, std = self.spk2meanstd[sample['utt_id'][:3]]
        sample['feat'] = ((sample['feat'] - mean) / std).astype(np.float32)
        return sample

def compute_spk_mean_std(dataset, save_file=None):
    ''' 
    Computes the mean and standard deviation for all acoustic frames for each 
    speaker in the dataset.
    '''
    spk2count = {}
    spk2sum = {}
    spk2sqsum = {}
    for bucket in dataset.buckets.keys():
        for utt in tqdm.tqdm(dataset.buckets[bucket]):
            idx = dataset.utt2idx[utt]
            feat = dataset[idx]['feat']
            spk = utt[:3]
            if spk not in spk2sum.keys():
                spk2count[spk] = feat.shape[0]
                spk2sum[spk] = np.sum(feat, axis=0)
                spk2sqsum[spk] = np.sum(feat ** 2, axis=0)
            else:
                spk2count[spk] += feat.shape[0]
                spk2sum[spk] += np.sum(feat, axis=0)
                spk2sqsum[spk] += np.sum(feat ** 2, axis=0)

    spk2meanstd = {}
    for spk in spk2sum.keys():
        mean = spk2sum[spk] / spk2count[spk]
        std = np.sqrt(spk2sqsum[spk] / spk2count[spk] - (mean ** 2))
        spk2meanstd[spk] = np.stack((mean, std), axis=0).tolist()

    if save_file is not None:
        utils.safe_json_dump(spk2meanstd, save_file)

    return spk2meanstd

if __name__=='__main__':
    datadir = '/share/data/speech/Data/dyunis/data/wsj_espnet'
    tok_file = 'lang_1char/train_si284_units.txt'
    load_dir = 'buckets'

    jsons = {'train': 'dump/train_si284/deltafalse/data.json',
             'dev': 'dump/test_dev93/deltafalse/data.json',
             'test': 'dump/test_eval92/deltafalse/data.json'}

    for split in jsons.keys():
        dataset = datasets.ESPnetBucketDataset(os.path.join(datadir,
                                                            jsons[split]),
                                               os.path.join(datadir,
                                                            tok_file),
                                               n_buckets=10)

        json_path = os.path.join(datadir, load_dir, 
                                f'{split}_stats/spk2meanstd.json')
        compute_spk_mean_std(dataset, save_file=json_path)

        with open(json_path, 'r') as f:
            spk2meanstd = json.load(f)

        for spk in spk2meanstd.keys():
            spk2meanstd[spk] = np.array(spk2meanstd[spk])

        spk_norm = SpeakerNormalize(spk2meanstd)
