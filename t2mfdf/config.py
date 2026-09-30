"""Validated experimental configuration. Unknown keys fail instead of being ignored."""
import json
import math
from pathlib import Path

DEFAULT = dict(d_model=128, heads=8, patch_length=32, patch_stride=16,
    dropout=.1, num_classes=4, modality='multimodal', route='prose',
    text_backend='bert', bert_model='google-bert/bert-base-uncased',
    backbone='bert', text_scope='patch', max_text_length=256, bert_chunk_size=64,
    source_tokens=2000, seed=42, batch_size=64, epochs=5, patience=2,
    lr=.0001, weight_decay=0., train_fraction=1., target=None,
    paper_class_ratios=False, text_context_layers=1, text_context_ff=512,
    num_workers=0, gradient_clip=1., accumulation_steps=1, amp=False,
    cache_tokens=True, split_mode='group', model_name='t2mfdf', use_token_prototypes=True)


def load_config(path=None, **overrides):
    config = dict(DEFAULT)
    if path:
        config.update(json.loads(Path(path).read_text(encoding='utf-8')))
    config.update({k: v for k, v in overrides.items() if v is not None})
    validate(config)
    return config


def validate(c):
    unknown = set(c) - set(DEFAULT) - {'bert_config'}
    if unknown:
        raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    for key in ('lr','weight_decay','train_fraction','dropout','gradient_clip'):
        if not math.isfinite(c[key]):
            raise ValueError(f'{key} must be finite')
    for key in ('d_model', 'heads', 'patch_length', 'patch_stride', 'num_classes',
                'max_text_length', 'bert_chunk_size', 'source_tokens', 'batch_size',
                'epochs', 'patience', 'text_context_ff', 'accumulation_steps'):
        if not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if c['d_model'] % 2 or c['d_model'] % c['heads']:
        raise ValueError('d_model must be even and divisible by heads')
    if not 8 <= c['patch_length'] <= 1024 or c['patch_stride'] > c['patch_length']:
        raise ValueError('Patch size must be 8..1024; stride must not exceed patch size')
    for key, values in dict(modality=('time','text','multimodal'), route=('prose','figure'),
            text_backend=('tiny','bert'), backbone=('bert','gpt2'), text_scope=('patch','window'),
            split_mode=('group','window'), model_name=('t2mfdf','wdcnn','dlinear')).items():
        if c[key] not in values:
            raise ValueError(f'{key} must be one of {values}')
    if not 0 < c['train_fraction'] <= 1 or not 0 <= c['dropout'] < 1 or c['lr'] <= 0:
        raise ValueError('Invalid train_fraction/dropout/lr')
    if c['num_workers'] < 0 or c['text_context_layers'] < 0 or c['weight_decay'] < 0 or c['gradient_clip'] <= 0:
        raise ValueError('Invalid worker/layer count, weight decay or gradient clip')
    if c['model_name'] != 't2mfdf' and c['modality'] != 'time':
        raise ValueError('Signal baselines require modality=time')
