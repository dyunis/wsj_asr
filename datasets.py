import os
import json

import torch
import numpy as np
import kaldiio

import utils

class ESPnetDataset(torch.utils.data.Dataset):
    '''
    A PyTorch dataset reorganizing the JSON that results from ESPnet.
    '''
    def __init__(self, json_file, tok_file, transform=None):
        self.json_file = json_file
        with open(json_file, 'r') as f:
            self.json = json.load(f)['utts']
        self.idx2utt = list(self.json)
        self.utt2idx = {utt: idx for idx, utt in enumerate(self.idx2utt)}
        self.transform = transform

        self.idx2tok = {}
        with open(tok_file, 'r') as f:
            for line in f:
                line = line.split()
                self.idx2tok[int(line[1])] = line[0]

    def __getitem__(self, idx):
        utt_id = self.idx2utt[idx]
        ark_fn, label = (self.json[utt_id]['input'][0]['feat'], 
                         self.json[utt_id]['output'][0]['tokenid'])
        label = [int(s) for s in label.split()]
        
        feat = kaldiio.load_mat(ark_fn)

        sample = {'utt_id': utt_id, 'feat': feat, 'label': label}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.idx2utt)

class ESPnetBucketDataset(ESPnetDataset):
    '''
    Like ESPnetDataset, but buckets the data into {n_buckets} according to
    length so that similar length utterances are in the same bucket.
    '''
    def __init__(self, json_file, tok_file, load_dir=None, save_dir=None, 
                 n_buckets=1, transform=None):
        super().__init__(json_file, tok_file, transform)
        utt_ids = self.json.keys()
        feat_lens = {utt_id: self.json[utt_id]['input'][0]['shape'][0] 
                     for utt_id in utt_ids}

        if load_dir is not None:
            self.n_buckets, self.buckets, self.utt2bucket = load_buckets(load_dir)
        else:
            self.buckets, self.utt2bucket = bucket_dataset(utt_ids, feat_lens,
                                                           n_buckets)
            if save_dir is not None:
                save_buckets(self.buckets, save_dir)

            self.n_buckets = n_buckets

def bucket_dataset(utt_ids, feat_lens, n_buckets):
    '''
    Buckets a list of utterances into buckets by length.
    '''
    buckets = {}
    utt2bucket = {}
    utt_and_len = []
    for utt_id in utt_ids:
        utt_and_len.append((utt_id, feat_lens[utt_id]))
    
    # sort the utterances according to length
    utt_and_len = sorted(utt_and_len, key=lambda tupl: tupl[1])
    utts = [tupl[0] for tupl in utt_and_len] 

    # divide the utterances into buckets
    per_bucket = len(utts) // n_buckets
    for i in range(n_buckets):
        buckets[i] = utts[i * per_bucket:(i + 1) * per_bucket]
    
        for utt in buckets[i]:
            utt2bucket[utt] = i

    # add the remainder to the last bucket
    if n_buckets * per_bucket < len(utts):
        buckets[n_buckets - 1] = (buckets[n_buckets - 1] + 
                                    utts[n_buckets * per_bucket:])

    return buckets, utt2bucket

def save_buckets(buckets, bucket_dir):
    '''
    Saves the bucket lists to a directory.
    '''
    utils.safe_makedirs(bucket_dir)

    import pdb; pdb.set_trace()
    for i in buckets.keys():
        fname = os.path.join(bucket_dir, str(i))
        with open(fname, 'w') as f:
            for utt in buckets[i]:
                f.write(utt + '\n')

def load_buckets(bucket_dir):
    '''
    Loads the bucket lists from a directory.
    '''
    if not os.path.exists(bucket_dir):
        raise OSError(f'Path to load {bucket_dir} does not exist')

    buckets = {} 
    utt2bucket = {}

    bucket_files = [os.path.join(bucket_dir, f) for f in os.listdir(bucket_dir) 
                    if os.path.isfile(os.path.join(bucket_dir, f))]

    for i, bucket_file in enumerate(bucket_files):
        buckets[i] = []
        with open(bucket_file, 'r') as f:
            for line in f:
                utt = line.strip()
                buckets[i].append(utt)
                utt2bucket[utt] = i

    return len(bucket_files), buckets, utt2bucket

class BucketBatchSampler(torch.utils.data.Sampler):
    '''
    A PyTorch Sampler for sampling minibatches from a bucketed dataset via
    stratified sampling.
    '''
    def __init__(self, shuffle, batch_size, utt2idx, buckets, seed=0):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.utt2idx = utt2idx
        self.n_buckets = len(buckets.keys())
        self.bucket2idx = []
        for i in buckets.keys():
            self.bucket2idx.append(np.array([self.utt2idx[utt]
                                             for utt in buckets[i]]))

        self.length = sum([len(bucket) for bucket in self.bucket2idx])
        self.init_num_batches()

        self.start_idx = [0 for i in range(self.n_buckets)]

    def init_num_batches(self):
        '''
        Creates the list of which bucket to sample from, where the index for a
        bucket appears as many times as there are batches to sample from it.
        '''
        num_batches = [len(bucket) for bucket in self.bucket2idx]
        num_batches = [int(np.ceil(num / self.batch_size)) 
                       for num in num_batches]

        # populate a list with batch_nums of each bucket
        self.bucket_list = np.array([i for i in range(self.n_buckets) 
                                     for j in range(num_batches[i])])
    
    def reset_epoch(self):
        '''
        Shuffles the sampler's bucket lists.
        '''
        self.start_idx = [0 for j in range(self.n_buckets)]
        if self.shuffle:
            self.bucket_list = np.random.permutation(self.bucket_list)
            for i in range(len(self.bucket2idx)):
                self.bucket2idx[i] = np.random.permutation(self.bucket2idx[i])

    def __iter__(self):
        '''
        Yields the list of batches to sample by sampling a bucket, and then a
        set of indices from that bucket for every batch.
        '''
        self.reset_epoch()
        for i, bucket in enumerate(self.bucket_list):
            start = self.start_idx[bucket]
            batch_idxs = self.bucket2idx[bucket][start:start+self.batch_size]

            self.start_idx[bucket] = start + self.batch_size

            yield batch_idxs.tolist()

    def __len__(self):
        return len(self.bucket_list)

def collate(minibatch):
    '''
    Aggregates the data in a batch and pads the acoustic features and labels 
    to equal length.
    '''
    idx2utt, feats, labels = [], [], []
    for i in range(len(minibatch)):
        idx2utt.append(minibatch[i]['utt_id'])
        feats.append(torch.tensor(minibatch[i]['feat'])) 
        labels.append(torch.tensor(minibatch[i]['label'], dtype=torch.int32))
    batch = {}
    batch['utt_id'] = idx2utt
    batch['feat'] = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
    batch['feat_lens'] = torch.tensor([feat.shape[0] for feat in feats], 
                                      dtype=torch.int32)
    batch['label'] = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
    batch['label_lens'] = torch.tensor([label.shape[0] for label in labels],
                                       dtype=torch.int32)
    return batch

def collate_cuda(minibatch):
    '''
    Sends aggregated batch data to the GPU. 
    '''
    batch = collate(minibatch)
    batch['feat'] = batch['feat'].cuda()
    batch['label'] = batch['label'].cuda()
    batch['feat_lens'] = batch['feat_lens'].cuda()
    batch['label_lens'] = batch['label_lens'].cuda()
    return batch

if __name__=='__main__':
    datadir = '/share/data/speech/Data/dyunis/data/wsj_espnet'
    json_file = 'dump/train_si284/deltafalse/data.json'
    tok_file = 'lang_1char/train_si284_units.txt'
    savedir = '/share/data/speech/Data/dyunis/data/wsj_espnet/buckets'
    json_file = os.path.join(datadir, json_file)
    tok_file = os.path.join(datadir, tok_file)

    dataset = ESPnetBucketDataset(json_file, tok_file, save_dir=savedir,
                                  n_buckets=10)
