import numpy as np

def greedy_ctc_decode(log_prob, blank_idx=0, zero_infinity=False):
    '''
    Greedily decodes a sequence of log probabilities computed via CTC
    (Connectionist Temporal Classification) loss: first duplicate indices are 
    ignored, then indices corresponding to a blank token are removed.
    '''
    if zero_infinity:
        log_prob = np.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=-1e10)

    idxs = np.argmax(log_prob, axis=-1) # assuming time idx is 1
    decoded = []
    
    prev_idx = -1 # this will be the blank idx
    for i in range(len(idxs)):
        if idxs[i] == blank_idx:
            prev_idx = -1
            continue
        elif idxs[i] == prev_idx:
            continue
        else:
            decoded.append(idxs[i])
        prev_idx = idxs[i]

    return decoded

def batch_greedy_ctc_decode(log_probs, blank_idx=0, zero_infinity=False, 
                            to_remove=-1):
    '''
    Greedily decodes a batch of log probabilities computed via CTC loss.
    {log_probs} has shape array is [bsize, seq_len, vocab_size].
    
    Returns array of shape [bsize, seq_len] where all values of {to_remove} will
    be stripped.
    '''
    if zero_infinity:
        log_probs = np.nan_to_num(log_probs, nan=0.0, posinf=0.0, neginf=-1e10)

    idxs = np.argmax(log_probs, axis=-1)
    decoded = idxs.copy()
    
    prev_idxs = to_remove * np.ones(idxs.shape[0])
    for i in range(idxs.shape[1]):
        batch = idxs[:, i]
        decode_batch = decoded[:, i]

        decode_batch[batch == blank_idx] = to_remove 
        decode_batch[batch == prev_idxs] = to_remove
        prev_idxs = batch

    return decoded, to_remove

def compute_words(char_idxs, idx2char, space_idx=18):
    '''
    Computes the word sequence corresponding to a list of character indices,
    where {idx2char} is the dictionary for the mapping.
    '''
    chars = [idx2char[idx] for idx in char_idxs]
    words = ''.join(chars)
    words = words.replace(idx2char[space_idx], ' ')
    words = words.split()
    return words

def compute_cer_wer(log_probs, label, idx2tok):
    '''
    Computes character error rate and word error rate for each sequence decoded
    from a batch of log probabilities predicted by a model.
    '''
    batch_decoded, to_remove = batch_greedy_ctc_decode(log_probs, 
                                                       zero_infinity=True)

    cer, wer = [], []
    for i in range(log_probs.shape[0]):
        batch_dec = batch_decoded[i, :]
        batch_dec = batch_dec[batch_dec != to_remove]
        pred_words = compute_words(batch_dec, idx2tok)

        batch_lbl = label[i, :]
        label_chars = batch_lbl[batch_lbl != 0] # remove padding
        label_words = compute_words(label_chars, idx2tok)

        char_errors = levenshtein(batch_dec, label_chars)
        word_errors = levenshtein(pred_words, label_words)
        cer.append(char_errors / len(label_chars))
        wer.append(word_errors / len(label_words))

    return cer, wer

def levenshtein(pred, label):
    '''
    Computes Levenshtein distance between a predicted and a label set of tokens.
    '''
    d = np.zeros((len(pred) + 1, len(label) + 1), dtype=np.int)
    for i in range(d.shape[0]):
        d[i, 0] = i

    for j in range(d.shape[1]):
        d[0, j] = j

    for i in range(1, d.shape[0]):
        for j in range(1, d.shape[1]):
            insert = d[i - 1, j] + 1
            delete = d[i, j - 1] + 1
            sub = d[i - 1, j - 1] + int(pred[i - 1] != label[j - 1])

            d[i, j] = min(delete, insert, sub)

    return d[-1, -1]

if __name__=='__main__':
    a = np.array([[1, 2, 3], [3, 2, 1], [1, 2, 3]])
    print(greedy_ctc_decode(a))
    a = np.array([[1, 2, 3], [3, 2, 1], [1, 2, 3]])
    a = a[None, ...]
    decoded, to_rm = batch_greedy_ctc_decode(a)
    print(decoded[decoded != to_rm])

    a = np.array([[3, 2, 1], [3, 2, 1], [1, 2, 3]])
    print(greedy_ctc_decode(a))
    a = np.array([[3, 2, 1], [3, 2, 1], [1, 2, 3]])
    a = a[None, ...]
    decoded, to_rm = batch_greedy_ctc_decode(a)
    print(decoded[decoded != to_rm])

    a = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    print(greedy_ctc_decode(a))
    a = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    a = a[None, ...]
    decoded, to_rm = batch_greedy_ctc_decode(a)
    print(decoded[decoded != to_rm])

    pred = ['o', 'this', 'is']
    label = ['this', 'is']
    print(levenshtein(pred, label))
    
