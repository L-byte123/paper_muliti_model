"""Prepare the selected 16 CWRU recordings using the shared importer.

Normal-record sampling rate is an explicit, recorded assumption, not inferred
from array length. No resampling or GPU training is performed.
"""
import argparse
import csv
import json
from pathlib import Path

from t2mfdf.prepare import prepare
from t2mfdf.inspect_data import inspect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--normal-fs', type=int, choices=(12000, 48000), required=True)
    args = parser.parse_args()
    raw = args.raw.resolve()
    out = args.output_dir.resolve()
    # A fresh directory avoids overwriting datasets or mixing different protocols.
    if out.exists():
        raise FileExistsError(f'Choose a new output directory: {out}')
    rows = []
    for label, ids in [(0, range(97, 101)), (1, range(105, 109)),
                       (2, range(130, 134)), (3, range(118, 122))]:
        for number in ids:
            path = raw / f'{number}.mat'
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append(dict(path=str(path), key=f'X{number:03d}_DE_time',
                             label=label, domain='CWRU',
                             fs=args.normal_fs if label == 0 else 12000,
                             group=f'CWRU_record_{number}'))
    out.mkdir(parents=True)
    manifest = out / 'manifest.csv'
    with manifest.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    protocol = dict(normal_fs=args.normal_fs, fault_fs=12000,
                    normal_fs_status='working assumption; not independently confirmed by file metadata',
                    normal_fs_reference='https://raw.githubusercontent.com/ivarejao/vibdata/master/vibdata/raw/CWRU/CWRU.csv',
                    window_length=1024, stride=1024, seed=42,
                    scope='16-record subset; not full paper reproduction',
                    group_unit='recording; not independent physical bearing',
                    caveat='Different loads can share a bearing. Recording-disjoint results do not establish bearing-disjoint generalization. Sampling-rate assumption must be resolved before reporting frequency-based results.')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    reports = {}
    for name, snr in [('cwru_clean.npz', None), ('cwru_snr10.npz', 10)]:
        path = out / name
        prepare(manifest, path, stride=1024, snr=snr, seed=42)
        reports[name] = inspect(path)
        print(f'{name}: {reports[name]["samples"]} windows', flush=True)
    (out / 'inspection.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print(f'Completed: {out}')


if __name__ == '__main__':
    main()
