import os
import shutil
import logging
import functools
import time
import inspect
import operator
import json

import torch

def ilen(iterator):
    '''
    Computes the length of an iterator without first constructing a list.
    '''
    return functools.reduce(increment_first, iterator, 0)

def increment_first(a, b):
    '''
    Increments the first argument.
    '''
    return a + 1

def timeit(function):
    '''
    Wraps a function to print the time it takes to execute.
    '''
    @functools.wraps(function)
    def timed(*args, **kwargs):
        start = time.time()
        result = function(*args, **kwargs)
        end = time.time()
        print(f'function name: {function.__name__}, ' +
              f'args: {args}, kwargs: {kwargs}')
        print(f'Call took {end-start:.6f} seconds\n')
        return result
    return timed

def debug(function):
    '''
    Wraps a function to call the debugger before executing.
    '''
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        import pdb; pdb.set_trace()
        return function(*args, **kwargs)
    return wrapped

def lineinfo(function):
    '''
    Wraps a function to give code location information.
    '''
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        print(f'filename: {inspect.stack()[1][1]}',
              f'line number: {inspect.stack()[1][2]}',
              f'function name: {function.__name__}') 
        return function(*args, **kwargs)
    return wrapped

def safe_copytree(src, tgt):
    '''
    Copies a directory tree if {src} exists and {tgt} doesn't.
    '''
    if not os.path.exists(src):
        raise OSError(f'Path to copy {src} does not exist')

    if os.path.exists(tgt):
        raise OSError(f'Target directory {tgt} already exists')

    try:
        shutil.copytree(src, tgt)

    except OSError as e:
        if e.errno == errno.ENOSPC:
            logging.info('Copying failed (no space on disk).')
            logging.info(f'Removing {tgt}')
            if os.path.exists(tgt):
                shutil.rmtree(tgt)
            raise

def safe_rmtree(tgt):
    '''
    Removes a directory tree if it exists.
    '''
    if not os.path.exists(tgt):
        raise OSError(f'Path to remove {tgt} does not exist')

    shutil.rmtree(tgt)

def safe_makedirs(path):
    '''
    Makes a directory if it doesn't already exist.
    '''
    if os.path.exists(path):
        raise OSError(f'Path to save models {path} already exists.')

    os.makedirs(path)

def safe_json_dump(obj, save_file):
    '''
    Saves an object to a json if {save_file} isn't taken.
    '''
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))

    if os.path.exists(save_file):
        raise OSError(f'Path to save {save_file} already exists')

    with open(save_file, 'w') as f:
        json.dump(obj, f)

def param_size(module):
    '''
    Computes memory use in MB of parameters of a PyTorch module.
    '''
    params = module.parameters(recurse=True)
    mb = 0
    for param in params:
        mb += param.element_size() * functools.reduce(operator.mul, param.shape)
    return mb/1e6

def grad_has_nans(param):
    '''
    Checks if a PyTorch parameter gradient has nans.
    '''
    return torch.sum(torch.isnan(param.grad)) > 0
