"""Train, evaluate and predict. Run `python -m t2mfdf.run --help`."""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import load_data, split_groups, split_windows, SignalDataset
from .baselines import build_model
from .config import DEFAULT, validate
from .checkpoint import atomic_save, rng_state, restore_rng


def metrics(labels, predictions, classes):
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    tp = np.diag(confusion)
    precision = tp / np.maximum(confusion.sum(0), 1)
    recall = tp / np.maximum(confusion.sum(1), 1)
    f1 = 2*precision*recall / np.maximum(precision+recall, 1e-12)
    present = confusion.sum(1) > 0
    return dict(accuracy=float(tp.sum()/max(confusion.sum(), 1)),
                macro_f1=float(f1.mean()), weighted_f1=float((f1*confusion.sum(1)).sum()/max(confusion.sum(), 1)),
                present_class_macro_f1=float(f1[present].mean()) if present.any() else 0.,
                per_class_f1=f1.tolist(), confusion_matrix=confusion.tolist())


def tokenizer_for(config, path=None):
    if config['modality'] == 'time' or config['text_backend'] == 'tiny':
        return None
    os.environ.setdefault('HF_HOME', str(Path('data/hf_cache').resolve()))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(path) if path else config['bert_model'])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    return tokenizer


def loader(data, indices, config, tokenizer, shuffle=False):
    return DataLoader(SignalDataset(data, indices, config, tokenizer),
                      batch_size=config['batch_size'], shuffle=shuffle,
                      num_workers=config.get('num_workers', 0))


def forward_batch(model, batch, device):
    labels = batch['label'].to(device)
    inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
    return model(**inputs), labels


@torch.no_grad()
def evaluate(model, batches, device):
    model.eval()
    total, count, labels, predictions = 0., 0, [], []
    for batch in batches:
        logits, y = forward_batch(model, batch, device)
        total += torch.nn.functional.cross_entropy(logits, y, reduction='sum').item()
        count += len(y)
        labels.extend(y.cpu().tolist())
        predictions.extend(logits.argmax(-1).cpu().tolist())
    if not count:
        raise ValueError('Empty evaluation split')
    result = metrics(labels, predictions, model.config['num_classes'])
    result['loss'], result['samples'] = total/count, count
    return result


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def balance_indices(data, indices, seed):
    """Downsample independently inside each split/domain; never duplicate examples."""
    rng = np.random.default_rng(seed)
    chosen = []
    for domain in np.unique(data['domains'][indices]):
        subset = indices[data['domains'][indices] == domain]
        classes = sorted(np.unique(data['y'][subset]).tolist())
        if classes not in ([0,1,2], [0,1,2,3]):
            raise ValueError(f'{domain}: paper ratios need labels 0/1/2 or 0/1/2/3')
        ratios = [1,1,1] if len(classes) == 3 else [1,3,3,3]
        pools = [subset[data['y'][subset] == label] for label in classes]
        unit = min(len(pool)//ratio for pool,ratio in zip(pools,ratios))
        if unit == 0:
            raise ValueError(f'{domain}: too few examples for paper class ratios')
        for pool, ratio in zip(pools, ratios):
            chosen.extend(rng.choice(pool, unit*ratio, replace=False).tolist())
    return np.array(chosen, dtype=np.int64)


def train(data_path, output, config, device, resume=False):
    validate(config)
    output = Path(output)
    saved = None
    data_sha = digest(data_path)
    if resume:
        saved = torch.load(output/'last.pt', map_location='cpu', weights_only=True)
        if saved['data_sha256'] != data_sha:
            raise ValueError('Resume requires exactly the original dataset')
        for key, value in saved['config'].items():
            if key not in ('epochs', 'bert_config') and config.get(key) != value:
                raise ValueError(f'Resume cannot change {key}; use a new experiment')
        config = dict(saved['config'], epochs=config['epochs'])
        if config['epochs'] < saved['epoch']:
            raise ValueError('Requested epochs precede the resume checkpoint')
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    data = load_data(data_path)
    if data['y'].max() >= config['num_classes']:
        raise ValueError('num_classes does not cover labels')
    if not 0 < config['train_fraction'] <= 1:
        raise ValueError('train_fraction must be in (0,1]')
    source = np.arange(len(data['x']))
    target = np.array([], dtype=np.int64)
    if config['target']:
        target = np.flatnonzero(data['domains'] == config['target'])
        source = np.flatnonzero(data['domains'] != config['target'])
        if len(target) == 0 or len(source) == 0:
            raise ValueError('Target and source domains must both exist')
    source_data = {k: v[source] for k, v in data.items()}
    splitter = split_groups if config['split_mode'] == 'group' else split_windows
    train_idx, val_idx, test_idx = [source[i] for i in splitter(source_data, config['seed'])]
    if config['paper_class_ratios']:
        train_idx, val_idx, test_idx = [balance_indices(data, idx, config['seed']+j)
                                       for j, idx in enumerate([train_idx,val_idx,test_idx])]
        if len(target):
            target = balance_indices(data, target, config['seed']+3)
    if config['train_fraction'] < 1:
        rng = np.random.default_rng(config['seed'])
        n = int(len(train_idx) * config['train_fraction'])
        if n < config['num_classes']:
            raise ValueError('Requested fraction too small to cover classes; use real data or larger fraction')
        # Fixed total count, approximately stratified, no oversampling above requested fraction.
        selected = []
        for label in np.unique(data['y'][train_idx]):
            pool = train_idx[data['y'][train_idx] == label]
            count = max(1, int(len(pool)*config['train_fraction']))
            selected.extend(rng.choice(pool, count, replace=False).tolist())
        if len(selected) > n:
            raise ValueError('Fraction cannot retain all classes with requested counts')
        rest = np.setdiff1d(train_idx, selected)
        selected.extend(rng.choice(rest, n-len(selected), replace=False).tolist())
        train_idx = np.array(selected)
    if set(np.unique(data['y'][train_idx])) != set(range(config['num_classes'])):
        raise ValueError('Training split must contain each class; choose a consistent class set')
    if (output/'best.pt').exists() and not resume:
        raise FileExistsError('Output already has a checkpoint; choose a new --output')
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output/'splits.npz', train=train_idx, val=val_idx, test=test_idx, target=target)
    tokenizer = tokenizer_for(config, output/'tokenizer' if resume and config['text_backend'] != 'tiny' else None)
    if tokenizer is not None:
        tokenizer.save_pretrained(output/'tokenizer')
    model = build_model(config, load_pretrained=not resume).to(device)
    config = model.config
    (output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    batches = loader(data, train_idx, config, tokenizer, True)
    validation = loader(data, val_idx, config, tokenizer)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=config['lr'], weight_decay=config['weight_decay'])
    use_amp = config['amp'] and str(device).startswith('cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    best, bad, history = float('inf'), 0, []
    start = 0
    if saved:
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        best, bad, history, start = saved['best'], saved['bad'], saved['history'], saved['epoch']
        restore_rng(saved['rng'])
    for epoch in range(start, config['epochs']):
        if bad >= config['patience']:
            break
        model.train()
        total, seen = 0., 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(batches):
            with torch.autocast(device_type='cuda' if use_amp else 'cpu', enabled=use_amp):
                logits, y = forward_batch(model, batch, device)
                loss = torch.nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            # Sample-weighted accumulation also handles the last short minibatch.
            group_start = (step // config['accumulation_steps']) * config['accumulation_steps']
            group_count = min(config['accumulation_steps'] * config['batch_size'],
                              len(train_idx) - group_start * config['batch_size'])
            scaler.scale(loss * len(y) / group_count).backward()
            if (step+1) % config['accumulation_steps'] == 0 or step+1 == len(batches):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total += loss.item()*len(y)
            seen += len(y)
        result = evaluate(model, validation, device)
        record = dict(epoch=epoch+1, train_loss=total/seen, validation=result)
        history.append(record)
        print(json.dumps(record), flush=True)
        if result['loss'] < best:
            best, bad = result['loss'], 0
            atomic_save(dict(config=config, state_dict=model.state_dict(), epoch=epoch+1,
                             data_sha256=data_sha), output/'best.pt')
        else:
            bad += 1
        (output/'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        atomic_save(dict(config=config, state_dict=model.state_dict(), epoch=epoch+1,
                         optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                         best=best, bad=bad, history=history, rng=rng_state(),
                         data_sha256=data_sha), output/'last.pt')
        if bad >= config['patience']:
            break
    checkpoint = torch.load(output/'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    report = dict(test=evaluate(model, loader(data, test_idx, config, tokenizer), device),
                  data_sha256=data_sha, best_epoch=checkpoint['epoch'],
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  counts=dict(train=len(train_idx), val=len(val_idx), test=len(test_idx), target=len(target)),
                  torch_version=str(torch.__version__), device=str(device),
                  smoke_test_only=config['text_backend']=='tiny', config=config,
                  split_mode=config['split_mode'])
    report['software'] = dict(python=__import__('platform').python_version(),
                              numpy=np.__version__, cuda=torch.version.cuda)
    if str(device).startswith('cuda'):
        report['software']['gpu'] = torch.cuda.get_device_name(device)
    if len(target):
        report['transfer'] = evaluate(model, loader(data, target, config, tokenizer), device)
    (output/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


def restore(checkpoint, device):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model = build_model(saved['config'], load_pretrained=False).to(device)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    tokenizer = tokenizer_for(saved['config'], Path(checkpoint).parent/'tokenizer')
    return model, tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['train', 'evaluate', 'predict'])
    p.add_argument('--data', required=True, help='NPZ dataset, or 1D NPY for predict')
    p.add_argument('--output', default='runs/main')
    p.add_argument('--config')
    p.add_argument('--checkpoint')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--fs', type=float, default=12000)
    p.add_argument('--split', choices=['test','val','target','all'], default='test')
    p.add_argument('--epochs', type=int)
    p.add_argument('--modality', choices=['multimodal','time','text'])
    p.add_argument('--target', choices=['HIT','JNU','PU','CWRU'])
    p.add_argument('--train-fraction', type=float)
    p.add_argument('--seed', type=int)
    p.add_argument('--resume', action='store_true', help='Resume output/last.pt at epoch boundary')
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    if a.command == 'train':
        config = dict(DEFAULT)
        if a.config:
            config.update(json.loads(Path(a.config).read_text(encoding='utf-8')))
        for key in ['epochs', 'modality', 'target', 'train_fraction', 'seed']:
            if getattr(a, key) is not None:
                config[key] = getattr(a, key)
        train(a.data, a.output, config, a.device, a.resume)
    else:
        if not a.checkpoint:
            p.error('--checkpoint required')
        model, tokenizer = restore(a.checkpoint, a.device)
        if a.command == 'evaluate':
            data = load_data(a.data)
            if a.split == 'all':
                idx = np.arange(len(data['x']))
            else:
                report = json.loads((Path(a.checkpoint).parent/'metrics.json').read_text())
                if digest(a.data) != report['data_sha256']:
                    raise ValueError('Saved splits belong to different data; use --split all for external data')
                with np.load(Path(a.checkpoint).parent/'splits.npz') as splits:
                    idx = splits[a.split]
            print(json.dumps(evaluate(model, loader(data, idx, model.config, tokenizer), a.device), indent=2))
        else:
            x = np.load(a.data, allow_pickle=False).astype(np.float32)
            if x.shape != (1024,) or not np.isfinite(x).all() or a.fs <= 0:
                raise ValueError('Prediction requires finite 1D 1024-point signal and positive fs')
            data = dict(x=x[None], y=np.array([0]), fs=np.array([a.fs]))
            batch = next(iter(loader(data, np.array([0]), model.config, tokenizer)))
            with torch.no_grad():
                logits, _ = forward_batch(model, batch, a.device)
            print(json.dumps(dict(label=int(logits.argmax(-1).item()),
                                  probabilities=logits.softmax(-1)[0].cpu().tolist())))


if __name__ == '__main__':
    main()
