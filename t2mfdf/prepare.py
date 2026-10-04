"""Manifest-based importer; never infer fault labels from filenames."""
import argparse
import csv
import json
import hashlib
from pathlib import Path
import numpy as np
from .features import add_noise
from .rawio import read_signal


def prepare(manifest, output, length=1024, stride=1024, snr=10, seed=42, target_fs=None):
    if length < 8 or stride < 1:
        raise ValueError('Invalid length/stride')
    if snr is not None and not np.isfinite(snr):
        raise ValueError('SNR must be finite')
    if target_fs is not None and (not np.isfinite(target_fs) or target_fs <= 0 or int(target_fs) != target_fs):
        raise ValueError('target_fs must be a positive integer')
    if Path(output).exists():
        raise FileExistsError(f'Dataset already exists: {output}')
    rng = np.random.default_rng(seed)
    parts = {k: [] for k in ('x', 'y', 'groups', 'domains', 'fs')}
    provenance, seen_records = [], {}
    with open(manifest, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        path = Path(row['path'])
        if not path.is_absolute():
            path = Path(manifest).resolve().parent / path
        key = row.get('key', '')
        raw = read_signal(path, key, column=row.get('column') or None,
                          skiprows=row.get('skiprows') or 0, delimiter=row.get('delimiter') or None)
        fs, label = float(row['fs']), int(row['label'])
        if not np.isfinite(fs) or fs <= 0 or label < 0 or not row['domain'].strip():
            raise ValueError(f'{path}: invalid fs/label/domain')
        if len(raw) < length or not np.isfinite(raw).all():
            raise ValueError(f'{path}: short or nonfinite signal')
        original_samples, original_fs = len(raw), fs
        record_hash = hashlib.sha256(raw.tobytes()).hexdigest()
        if target_fs is not None and fs != target_fs:
            from math import gcd
            from scipy.signal import resample_poly
            if int(fs) != fs:
                raise ValueError('Resampling requires integer source sample rate')
            divisor = gcd(int(fs), int(target_fs))
            raw = resample_poly(raw, int(target_fs)//divisor, int(fs)//divisor)
            fs = float(target_fs)
        if len(raw) < length:
            raise ValueError(f'{path}: signal too short after resampling')
        windows = np.stack([raw[i:i+length] for i in range(0, len(raw)-length+1, stride)]).astype(np.float32)
        if snr is not None:
            windows = add_noise(windows, snr, rng)
        n = len(windows)
        # Same physical record must retain the same group even if exported into several files.
        group = row.get('group') or str(path.resolve())
        identity = (group, int(row['label']), row['domain'])
        if record_hash in seen_records:
            raise ValueError(f'{path}: duplicate signal record; do not import the same recording twice')
        seen_records[record_hash] = identity
        provenance.append(dict(path=str(path.resolve()),key=key,group=group,label=label,
                               domain=row['domain'],fs=fs,samples=len(raw),
                               original_fs=original_fs,original_samples=original_samples,
                               decoded_signal_sha256=record_hash,windows=n,
                               column=row.get('column') or None,skiprows=row.get('skiprows') or 0,
                               delimiter=row.get('delimiter') or None))
        parts['x'].append(windows)
        for k, value in [('y', int(row['label'])), ('groups', group), ('domains', row['domain']), ('fs', fs)]:
            parts[k].append(np.full(n, value))
    if not rows:
        raise ValueError('Empty manifest')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{k: np.concatenate(v) for k, v in parts.items()})
    Path(str(output) + '.json').write_text(json.dumps(dict(manifest=str(Path(manifest).resolve()),
        length=length, stride=stride, snr_db=snr, seed=seed,target_fs=target_fs,records=provenance), indent=2), encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest')
    p.add_argument('--output', required=True)
    p.add_argument('--stride', type=int, default=1024)
    p.add_argument('--snr', type=float, default=10)
    p.add_argument('--clean', action='store_true', help='Do not add noise')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--target-fs', type=int, help='Optional anti-aliased resampling rate before windowing')
    a = p.parse_args()
    if a.manifest:
        prepare(a.manifest, a.output, stride=a.stride, snr=None if a.clean else a.snr, seed=a.seed, target_fs=a.target_fs)
    else:
        p.error('--manifest required')
