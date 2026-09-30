"""Inspect a prepared dataset before committing GPU time to training."""
import argparse
import json
from collections import Counter
import numpy as np
from .data import load_data, split_groups
from .run import digest


def inspect(path):
    data = load_data(path)
    counts = []
    for domain in np.unique(data['domains']):
        for label in np.unique(data['y'][data['domains']==domain]):
            idx = (data['domains']==domain) & (data['y']==label)
            counts.append(dict(domain=str(domain),label=int(label),windows=int(idx.sum()),
                               groups=len(np.unique(data['groups'][idx])),
                               sample_rates=np.unique(data['fs'][idx]).tolist()))
    result = dict(sha256=digest(path),samples=len(data['y']),counts=counts,
                  amplitude_min=float(data['x'].min()),amplitude_max=float(data['x'].max()))
    try:
        splits = split_groups(data)
        result['group_split_samples'] = [len(s) for s in splits]
        result['group_split_ready'] = True
    except ValueError as exc:
        result['group_split_ready'] = False
        result['reason'] = str(exc)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.data),indent=2))
