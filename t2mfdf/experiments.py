"""Reproducible multi-seed Table I protocols, ablations and result aggregation."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from .config import load_config
from .run import train

TARGETS = ['HIT', 'JNU', 'PU', 'CWRU']
VARIANTS = {
    'full': {}, 'time': {'modality':'time'}, 'text': {'modality':'text'},
    'gpt2': {'backbone':'gpt2', 'bert_model':'openai-community/gpt2'},
    'figure': {'route':'figure'}, 'no-context': {'text_context_layers':0},
    'no-prototypes': {'use_token_prototypes':False},
    'wdcnn': {'model_name':'wdcnn','modality':'time'},
    'dlinear': {'model_name':'dlinear','modality':'time'},
}


def summarize(root):
    root = Path(root)
    rows = []
    for path in sorted(root.glob('*/fraction_*/dataset*/seed*/metrics.json')):
        result = json.loads(path.read_text(encoding='utf8'))
        relative = path.relative_to(root).parts
        for split in ('test', 'transfer'):
            if split not in result:
                continue
            rows.append(dict(variant=relative[0], fraction=relative[1][9:],
                dataset=relative[2], seed=relative[3][4:], split=split,
                accuracy=result[split]['accuracy'], macro_f1=result[split]['macro_f1'],
                present_class_macro_f1=result[split]['present_class_macro_f1'],
                samples=result[split]['samples'], data_sha256=result['data_sha256']))
    if not rows:
        raise ValueError('No completed experiments found')
    if len({r['data_sha256'] for r in rows}) != 1:
        raise ValueError('Cannot aggregate runs from different prepared datasets')
    with (root/'results.csv').open('w', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    for row in rows:
        key = (row['variant'],row['fraction'],row['dataset'],row['split'])
        groups.setdefault(key, []).append(row)
    summary = []
    for key, values in groups.items():
        entry = dict(zip(('variant','fraction','dataset','split'),key), runs=len(values))
        for metric in ('accuracy','macro_f1','present_class_macro_f1'):
            vector = [v[metric] for v in values]
            entry[metric+'_mean'] = float(np.mean(vector))
            entry[metric+'_std'] = float(np.std(vector, ddof=1)) if len(vector)>1 else None
        summary.append(entry)
    # A quadrilateral on four equally spaced radial axes has area .5*sum(adjacent products).
    # Preserve Table I ordering and do not claim the paper's unspecified normalization.
    areas = []
    for variant, fraction in sorted({(r['variant'],r['fraction']) for r in rows}):
        points = [next((r['macro_f1_mean'] for r in summary if r['variant']==variant and
                  r['fraction']==fraction and r['dataset']==f'dataset{i}' and r['split']=='transfer'),None)
                  for i in range(1,5)]
        if all(v is not None for v in points):
            areas.append(dict(variant=variant,fraction=fraction,
                raw_area=float(.5*np.dot(points,np.roll(points,1))),
                normalized_area=float(.25*np.dot(points,np.roll(points,1)))))
    payload = dict(summary=summary, transfer_radar_area=areas)
    (root/'summary.json').write_text(json.dumps(payload,indent=2),encoding='utf8')
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data')
    p.add_argument('--config',default='configs/paper.json')
    p.add_argument('--output',default='runs/experiments')
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--fractions',nargs='+',type=float,default=[1.,.01])
    p.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=['full'])
    p.add_argument('--targets',nargs='+',choices=TARGETS,default=TARGETS)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--summarize-only',action='store_true')
    p.add_argument('--threads',type=int,default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    if not a.summarize_only:
        if not a.data:
            p.error('--data required')
        from .data import load_data
        domains = set(load_data(a.data)['domains'].tolist())
        if domains != set(TARGETS):
            raise ValueError(f'Table I experiments require exactly four domains {TARGETS}; got {domains}')
        for variant in a.variants:
            for fraction in a.fractions:
                for target in a.targets:
                    for seed in a.seeds:
                        output = Path(a.output)/variant/f'fraction_{fraction:g}'/f'dataset{TARGETS.index(target)+1}'/f'seed{seed}'
                        config = load_config(a.config,seed=seed,target=target,
                            train_fraction=fraction,**VARIANTS[variant])
                        report_path = output/'metrics.json'
                        if report_path.exists() and a.resume:
                            previous = json.loads(report_path.read_text(encoding='utf8'))
                            from .run import digest
                            if previous['data_sha256'] != digest(a.data) or any(
                                    previous['config'].get(k) != v for k,v in config.items()):
                                raise ValueError(f'Existing experiment configuration differs: {output}')
                            continue
                        train(a.data,output,config,a.device,a.resume and (output/'last.pt').exists())
    print(json.dumps(summarize(a.output),indent=2))


if __name__ == '__main__':
    main()
