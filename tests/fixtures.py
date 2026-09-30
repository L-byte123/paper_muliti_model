"""Analytic fixtures for software tests, not research datasets."""
from pathlib import Path
import numpy as np
from t2mfdf.features import add_noise

def synthetic_fixture(output, seed=42):
    rng = np.random.default_rng(seed)
    parts = {k: [] for k in ('x', 'y', 'groups', 'domains', 'fs')}
    for domain in ['CWRU', 'PU', 'JNU', 'HIT']:
        for label in range(3 if domain in ['PU', 'HIT'] else 4):
            for group in range(10):
                t = np.arange(1024) / 12000
                x = np.sin(2*np.pi*(150 + 350*label)*t + rng.uniform(0, 6.28))
                x += .3 * np.sin(2*np.pi*50*t)
                if label:
                    x[::max(5, 31-label*5)] += label
                parts['x'].append(add_noise(x[None].astype(np.float32), 10, rng)[0])
                parts['y'].append(label)
                parts['groups'].append(f'{domain}-{label}-{group}')
                parts['domains'].append(domain)
                parts['fs'].append(12000.)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{k: np.asarray(v) for k, v in parts.items()})
