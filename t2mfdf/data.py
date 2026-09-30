"""Portable NPZ contract: x[N,1024], y[N], groups[N], domains[N], fs[N]."""
import numpy as np
import torch
from torch.utils.data import Dataset
from .features import patch_texts


def load_data(path, length=1024):
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k] for k in ('x', 'y', 'groups', 'domains', 'fs')}
    n = len(data['x'])
    if data['x'].shape != (n, length) or not np.isfinite(data['x']).all():
        raise ValueError(f'x must be finite [N,{length}]')
    if any(data[k].shape != (n,) for k in ('y', 'groups', 'domains', 'fs')):
        raise ValueError('Metadata must have shape [N]')
    if n == 0 or (data['fs'] <= 0).any() or not np.isfinite(data['fs']).all():
        raise ValueError('Empty data or invalid sample rate')
    if data['y'].dtype.kind not in 'iu' or (data['y'] < 0).any():
        raise ValueError('Labels must be nonnegative integers')
    for group in np.unique(data['groups']):
        idx = data['groups'] == group
        if len(np.unique(data['domains'][idx])) != 1 or len(np.unique(data['y'][idx])) != 1:
            raise ValueError('A physical recording group cannot span domains or labels')
    return data


def split_groups(data, seed=42):
    """Stratify independent recording groups by domain/label before splitting windows."""
    rng = np.random.default_rng(seed)
    buckets = {}
    for group in np.unique(data['groups']):
        idx = np.flatnonzero(data['groups'] == group)
        labels, domains = np.unique(data['y'][idx]), np.unique(data['domains'][idx])
        if len(labels) != 1 or len(domains) != 1:
            raise ValueError('Each recording group must have one label and domain')
        buckets.setdefault((str(domains[0]), int(labels[0])), []).append(idx)
    splits = [[], [], []]
    for key, groups in buckets.items():
        if len(groups) < 3:
            raise ValueError(f'{key}: need >=3 independent groups; refusing window leakage')
        rng.shuffle(groups)
        nval = max(1, round(len(groups) * .1))
        ntest = max(1, round(len(groups) * .1))
        ntrain = len(groups) - nval - ntest
        for target, chunk in zip(splits, (groups[:ntrain], groups[ntrain:ntrain+nval], groups[ntrain+nval:])):
            target.extend(chunk)
    return tuple(np.concatenate(s).astype(np.int64) for s in splits)


def split_windows(data, seed=42):
    """Paper-style window split; explicitly permits recording leakage, for sensitivity only."""
    rng = np.random.default_rng(seed)
    splits = [[], [], []]
    for domain in np.unique(data['domains']):
        for label in np.unique(data['y'][data['domains'] == domain]):
            indices = np.flatnonzero((data['domains'] == domain) & (data['y'] == label))
            if len(indices) < 3:
                raise ValueError('Need at least three windows per domain/class')
            rng.shuffle(indices)
            n = max(1, round(len(indices) * .1))
            for bucket, part in zip(splits, (indices[2*n:], indices[:n], indices[n:2*n])):
                bucket.extend(part.tolist())
    return tuple(np.asarray(s, dtype=np.int64) for s in splits)


class SignalDataset(Dataset):
    def __init__(self, data, indices, config, tokenizer=None):
        self.data, self.indices, self.config, self.tokenizer = data, indices, config, tokenizer
        self.token_cache = {}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        i = self.indices[item]
        signal = self.data['x'][i].astype(np.float32)
        result = {'signal': torch.from_numpy(signal), 'label': torch.tensor(int(self.data['y'][i]))}
        if self.config['modality'] != 'time':
            if i in self.token_cache:
                result.update(self.token_cache[i])
                return result
            texts = patch_texts(signal, self.data['fs'][i], self.config['patch_length'],
                                self.config['patch_stride'], self.config['text_scope'])
            if self.config['text_backend'] == 'tiny':
                # Offline smoke tests only: deterministic ASCII IDs, NOT pretrained language features.
                length = self.config['max_text_length']
                ids = np.zeros((len(texts), length), dtype=np.int64)
                for j, text in enumerate(texts):
                    values = [1] + [ord(c) % 125 + 3 for c in text[:length-2]] + [2]
                    ids[j, :len(values)] = values
                result.update(input_ids=torch.from_numpy(ids), attention_mask=torch.from_numpy((ids != 0).astype(np.int64)))
            else:
                encoded = self.tokenizer(texts, padding='max_length', truncation=False,
                                         max_length=self.config['max_text_length'])
                if max(map(len,encoded['input_ids'])) > self.config['max_text_length']:
                    raise ValueError('Diagnostic text exceeds max_text_length; increase it to avoid lost features')
                result.update(input_ids=torch.tensor(encoded['input_ids']),
                              attention_mask=torch.tensor(encoded['attention_mask']))
            if self.config.get('cache_tokens', False):
                self.token_cache[i] = {k: result[k] for k in ('input_ids', 'attention_mask')}
        return result
