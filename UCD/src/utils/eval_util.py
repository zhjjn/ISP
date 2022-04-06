"""
Helper functions for computing evaluation metrics.
"""
import numpy as np


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_performance(y_true, y_pred, config):
    y_true, y_pred = post_process(y_true, y_pred, config)
    return compute_seq_acc(y_true, y_pred)


def rm_idx(seq, idx):
    return [i for i in seq if i != idx]


def post_process(tgts, preds, config):
    # remove pad idx
    tgts = [rm_idx(tgt[:], config.PAD_IDX) for tgt in tgts]
    preds = [rm_idx(pred[:], config.PAD_IDX) for pred in preds]

    # remove end idx
    end_indices = [p.index(config.END_IDX)+1 if config.END_IDX in p else len(p) for p in preds]
    preds = [p[1:idx] for idx, p in zip(end_indices, preds)]
    tgts = [p[1:idx] for idx, p in zip(end_indices, tgts)]
    return tgts, preds


def check_if_seqs_identical(tar, pred):
    min_len = min([len(tar), len(pred)])
    if sum(np.equal(tar[:min_len], pred[:min_len])) == len(tar):
        return 1
    return 0


def compute_seq_acc(tars, preds):
    size = len(tars)
    a = 0
    for i in range(size):
        tar = tars[i]
        pred = preds[i]
        a += check_if_seqs_identical(tar, pred)

    return np.float32(a/size)
