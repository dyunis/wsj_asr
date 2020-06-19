import os
import argparse
import logging
import json
import itertools

import numpy as np
import torch
import tqdm

import datasets
import decoder
import utils
import models
import transforms

def main(args):
    '''
    Training and evaluation script for character-based CTC ASR on WSJ 
    dataset, pre-processed by ESPnet toolkit
    '''
    jsons = {'train': 'dump/train_si284/deltafalse/data.json',
             'dev': 'dump/test_dev93/deltafalse/data.json',
             'test': 'dump/test_eval92/deltafalse/data.json'}


    # if the temporary directory contains a json, we'll assume it's correct
    if not os.path.exists(os.path.join(args.temp_root, jsons['train'])):
        # copy the data for faster reading than NFS
        utils.safe_copytree(args.data_root, args.temp_root)

    # if model_dir is specified, and it doesn't contain the log file, make it
    log_file = os.path.join(args.model_dir, args.log_file)
    if args.model_dir is not None and not os.path.exists(log_file):
        utils.safe_makedirs(args.model_dir)

    logging.basicConfig(filename=log_file, filemode='a', level=logging.INFO)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = not args.nondeterm
    np.random.seed(args.seed)

    if not args.eval_only:
        utils.safe_json_dump(vars(args),
                             os.path.join(args.model_dir, 'args.json'))
        epoch_stats = train(args, jsons)
        utils.safe_json_dump(epoch_stats, os.path.join(args.model_dir,
                                                       'epoch_stats.json'))
    if args.eval_only:
        data_root, temp_root = args.data_root, args.temp_root
        test, cpu, seed, cleanup = args.test, args.cpu, args.seed, args.cleanup
        with open(os.path.join(args.model_dir, 'args.json'), 'r') as f:
            json_dict = json.load(f)
        args = argparse.Namespace(**json_dict)
        args.data_root, args.temp_root = data_root, temp_root
        args.test, args.cpu, args.seed, args.cleanup = test, cpu, seed, cleanup

    evaluate(args, jsons)

    if args.cleanup:
        utils.safe_rmtree(args.temp_root)

def train(args, jsons):
    '''
    Trains a model to do character-based ASR using CTC loss on WSJ.
    '''
    trainset, trainloader = make_dataset_dataloader(args, jsons, split='train')
    devset, devloader = make_dataset_dataloader(args, jsons, split='dev')

    model = models.LSTM(num_layers=args.n_layers, hidden_dim=args.hidden_dim,
                        bidirectional=(not args.unidir))

    use_gpu = not args.cpu and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    model.to(device)

    ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, 
                                 betas=(0.9, 0.98))

    global best_wer
    best_wer = np.inf

    tr_epoch = lambda ep: train_epoch(ep, args, trainset, trainloader, devset,
                                      devloader, model, ctc_loss, optimizer)
    stats = list(map(tr_epoch, range(args.n_epochs)))

    return stats

def train_epoch(epoch, args, trainset, trainloader, devset, devloader, model,
                ctc_loss, optimizer):
    '''
    Trains a model for a single epoch and returns metrics collected over the 
    epoch: training loss, development loss, development character error rate and
    development word error rate.
    '''
    stats = {}
    stats['epoch'] = epoch

    model.train()
    trainb = lambda batch: train_batch(batch, model, ctc_loss, optimizer)
    train_loss = list(map(trainb, tqdm.tqdm(trainloader)))

    stats['train_loss'] = np.mean(train_loss)

    dev_stats = evaluate_epoch(devset, devloader, model, ctc_loss)

    stats['dev_loss'] = dev_stats['loss']
    stats['dev_cer'] = dev_stats['cer']
    stats['dev_wer'] = dev_stats['wer']

    log_stats(f'Epoch: {epoch}', stats)

    if args.model_dir is not None:
        global best_wer
        best_wer = save_model(args, model, epoch, stats['dev_wer'], best_wer)

    return stats

def log_stats(title, data):
    '''
    Logs data to a log file or the console, where {data} is a dictionary.
    '''
    logging.info(title)
    logging.info('----')
    for key, val in data.items():
        logging.info(f'{key}: {val}')
    logging.info('\n')

def save_model(args, model, epoch, current_er, best_er):
    '''
    Saves a model according to a few different conditions on the epoch and the
    error rate of the model.
    '''
    if epoch % args.save_interval == 0:
        torch.save(model.state_dict(), os.path.join(args.model_dir, 
                                                    f'{epoch}.pt'))

    if epoch == args.n_epochs - 1:
        torch.save(model.state_dict(), os.path.join(args.model_dir, 
                                                    f'{epoch}.pt'))

    if current_er < best_er:
        torch.save(model.state_dict(), os.path.join(args.model_dir, 
                                                    f'best.pt'))
        best_er = current_er
        logging.info('saving best model')

    return best_er

def train_batch(batch, model, ctc_loss, optimizer):
    '''
    Updates a model over a single batch of the training data.
    '''
    optimizer.zero_grad()
    log_probs, embed = model(batch['feat'])
    loss = ctc_loss(log_probs.transpose(0, 1), batch['label'], 
                    batch['feat_lens'], batch['label_lens'])
    loss.backward()

    torch.nn.utils.clip_grad_norm(model.parameters(), 10.0)

    nans = filter(utils.grad_has_nans, model.parameters())
    if utils.ilen(nans) > 0:
        logging.info('Skipping training due to NaN in gradient')
        optimizer.zero_grad()

    optimizer.step()

    return float(loss)

def evaluate(args, jsons):
    '''
    Evaluates a model trained for character-based ASR with CTC loss on WSJ.
    '''
    model = models.LSTM(num_layers=args.n_layers, hidden_dim=args.hidden_dim,
                        bidirectional=(not args.unidir))

    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'best.pt')))

    use_gpu = not args.cpu and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    model.to(device)

    ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    splits = ['train', 'dev']
    if args.test:
        splits.append('test')

    for split in splits:
        dataset, dataloader = make_dataset_dataloader(args, jsons, split=split)
        stats = evaluate_epoch(dataset, dataloader, model, ctc_loss)
        log_stats(f'Final results on {split}', stats)

def evaluate_epoch(dataset, dataloader, model, ctc_loss):
    '''
    Evaluates a model on a particular dataset and returns the relevant metrics:
    loss, character error rate, and word error rate.
    '''
    model.eval()
    evalb = lambda batch: eval_batch(batch, model, ctc_loss, dataset.idx2tok)
    res = list(map(evalb, tqdm.tqdm(dataloader)))
    
    loss = (stat['loss'] for stat in res)

    cer = (stat['cer'] for stat in res)
    cer = itertools.chain.from_iterable(cer)

    wer = (stat['wer'] for stat in res)
    wer = itertools.chain.from_iterable(wer)

    stats = {}
    stats['loss'] = np.mean(list(loss))
    stats['cer'] = np.mean(list(cer))
    stats['wer'] = np.mean(list(wer))

    return stats

def eval_batch(batch, model, ctc_loss, idx2tok):
    '''
    Evaluates a model over a single batch.
    '''
    stats = {}
    log_probs, embed = model(batch['feat'])
    loss = ctc_loss(log_probs.transpose(0, 1), batch['label'], 
                    batch['feat_lens'], batch['label_lens'])
    stats['loss'] = float(loss)

    stats['cer'], stats['wer'] = decoder.compute_cer_wer(
                                     log_probs.cpu().detach().numpy(), 
                                     batch['label'].cpu().detach().numpy(),
                                     idx2tok)
    return stats

def make_dataset_dataloader(args, jsons, split='train'):
    '''
    Initializes a PyTorch Dataset and Dataloader according to the args.
    '''
    if split == 'train' and args.bucket_load_dir is not None:
        load_dir = os.path.join(args.temp_root, args.bucket_load_dir)
    else:
        load_dir = None

    dataset = datasets.ESPnetBucketDataset(os.path.join(args.temp_root, 
                                                        jsons[split]),
                                           os.path.join(args.temp_root,
                                                        args.tok_file),
                                           load_dir=load_dir,
                                           n_buckets=args.n_buckets)

    if args.normalize == 'utt':
        dataset.transform = transforms.Normalize()
    elif args.normalize == 'spk':
        spk_file = os.path.join(args.temp_root, args.bucket_load_dir,
                                f'{split}_stats/spk2meanstd.json')

        if os.path.isfile(spk_file):
            with open(spk_file, 'r') as f:
                spk2meanstd = json.load(f)

            for spk in spk2meanstd.keys():
                spk2meanstd[spk] = np.array(spk2meanstd[spk]) 
        else:
            spk2meanstd = transforms.compute_spk_mean_std(dataset)

        dataset.transform = transforms.SpeakerNormalize(spk2meanstd)

    if not args.cpu and torch.cuda.is_available():
        collate_fn = datasets.collate_cuda
    else:
        collate_fn = datasets.collate

    dataloader = torch.utils.data.DataLoader(
                     dataset, 
                     batch_sampler=datasets.BucketBatchSampler(
                                       shuffle=True, 
                                       batch_size=args.batch_size, 
                                       utt2idx=dataset.utt2idx, 
                                       buckets=dataset.buckets),
                     collate_fn=collate_fn)

    return dataset, dataloader

def check_valid(args):
    '''
    Checks that argument values are valid.
    '''
    assert os.path.isdir(args.data_root), \
           f'args.data_root: {args.data_root} must already exist'
    assert os.path.isfile(os.path.join(args.data_root, args.tok_file)), \
           f'args.tok_file: {args.tok_file} must already exist'
    if args.bucket_load_dir is not None:
        load_dir = os.path.join(args.temp_root, args.bucket_load_dir)
        assert os.path.isdir(load_dir), \
               (f'args.bucket_load_dir: {args.bucket_load_dir} must already ' +
                'exist')
    assert args.n_buckets > 0, \
           f'args.n_buckets: {args.n_buckets} must be > 0'
    
    assert args.n_layers > 0, \
           f'args.n_layers: {args.n_layers} must be > 0'
    assert args.hidden_dim > 0, \
           f'args.hidden_dim: {args.hidden_dim} must be > 0'

    assert args.n_epochs > 0, \
           f'args.n_epochs: {args.n_epochs} must be > 0'
    assert args.batch_size > 0, \
           f'args.batch_size: {args.batch_size} must be > 0'
    assert args.learning_rate > 0, \
           f'args.learning_rate: {args.learning_rate} must be > 0'
    assert args.save_interval > 0, \
           f'args.save_interval: {args.save_interval} must be > 0'
    
    if args.eval_only:
        assert os.path.exists(os.path.join(args.model_dir, 'args.json')), \
               f'args.model_dir: {args.model_dir} must contain args.json'
        assert os.path.exists(os.path.join(args.model_dir, 'best.pt')), \
               f'args.model_dir: {args.model_dir} must contain saved weights'

    if args.normalize is not None:
        assert args.normalize == 'utt' or args.normalize == 'spk', \
               (f'args.normalize: {args.normalize} must be one of [utt, spk]' +
                ' if not left blank')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='WSJ CTC character ASR model')

    # dataset args
    parser.add_argument('--eval_only',
                        action='store_true',
                        help='skip training step',
                        default=False)
    parser.add_argument('--data_root',
                        type=str,
                        help='data directory',
                        default='/share/data/speech/Data/dyunis/data/wsj_espnet')
    parser.add_argument('--temp_root',
                        help='temporary data directory',
                        default='/scratch/asr_tmp')
    parser.add_argument('--tok_file',
                        type=str,
                        help='token file (idx -> token)',
                        default='lang_1char/train_si284_units.txt')
    parser.add_argument('--log_file',
                        type=str,
                        help='file to log to',
                        default='log.log')
    parser.add_argument('--model_dir',
                        type=str,
                        help='directory to save models',
                        default='models')
    parser.add_argument('--bucket_load_dir',
                        type=str,
                        help='directory to load buckets from')
    parser.add_argument('--n_buckets',
                        type=int, 
                        help='number of buckets to split dataset into',
                        default=10)
    parser.add_argument('--test', 
                        help='record results on test set',
                        action='store_true',
                        default=False)

    # model args
    parser.add_argument('--hidden_dim',
                        type=int,
                        help='hidden dimension of the RNN acoustic model',
                        default=512)
    parser.add_argument('--n_layers', 
                        type=int,
                        help='number of layers of RNN acoustic model',
                        default=3)
    parser.add_argument('--unidir',
                        action='store_true',
                        help='use unidirectional RNN instead of bidirectional',
                        default=False)

    # optimizer args
    parser.add_argument('--n_epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=50)
    parser.add_argument('--batch_size',
                        type=int,
                        help='minibatch size for training',
                        default=16)
    parser.add_argument('--learning_rate',
                        type=float,
                        help='learning rate for training',
                        default=1e-4)
    parser.add_argument('--save_interval',
                        type=int,
                        help='save model every so many epochs',
                        default=10)
    
    # data preprocessing
    parser.add_argument('--normalize',
                        type=str,
                        help='type of normalization [utt, spk] to use',
                        default=None)


    # miscellaneous args
    parser.add_argument('--seed',
                        type=int,
                        help='seed for random number generators',
                        default=0)
    parser.add_argument('--cleanup',
                        action='store_true',
                        help='clean up temporary data at the end',
                        default=False)
    parser.add_argument('--nondeterm',
                        action='store_false',
                        help='nondeterministic behavior for cuda',
                        default=False)
    parser.add_argument('--cpu',
                        action='store_true',
                        help='only use cpu',
                        default=False)
    args = parser.parse_args()

    check_valid(args)
    main(args)
