"""Real pretrained BERT gradient/serialization validation, NOT an accuracy experiment.

Run after pip install -e .: python scripts/validate_pretrained.py --model MODEL_PATH
Analytic input is a test fixture only. This script never claims diagnostic accuracy.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from t2mfdf.config import load_config
from t2mfdf.run import tokenizer_for, loader, forward_batch
from t2mfdf.model import T2MFDF
from t2mfdf.checkpoint import atomic_save
from t2mfdf.run import restore


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='google-bert/bert-base-uncased')
    p.add_argument('--output',default='runs/pretrained_validation.json')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--route',choices=['prose','figure'],default='prose')
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    c = load_config(bert_model=a.model,batch_size=1,bert_chunk_size=4,cache_tokens=False,route=a.route)
    t = np.arange(1024)/12000
    data = dict(x=np.sin(2*np.pi*400*t)[None].astype(np.float32),
                y=np.array([1]),fs=np.array([12000]))
    tokenizer = tokenizer_for(c)
    model = T2MFDF(c).to(a.device).train()
    batch = next(iter(loader(data,np.array([0]),c,tokenizer)))
    logits,y = forward_batch(model,batch,a.device)
    loss = torch.nn.functional.cross_entropy(logits,y)
    loss.backward()
    gradients = {name: bool(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0)
                 for name,p in model.named_parameters() if p.requires_grad and
                 name in ('temporal.frequency.weight','vocabulary_mapping.weight',
                           'text_projection.weight','classifier.weight',
                           'text_context.layers.0.self_attn.in_proj_weight')}
    assert all(gradients.values())
    assert all(p.grad is None for p in model.bert.parameters())
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=c['lr'])
    optimizer.step()
    model.eval()
    with torch.no_grad():
        reference,_ = forward_batch(model,batch,a.device)
    checkpoint_dir = Path(a.output).parent/'pretrained_check'
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    tokenizer.save_pretrained(checkpoint_dir/'tokenizer')
    atomic_save(dict(config=model.config,state_dict=model.state_dict(),epoch=0),checkpoint_dir/'best.pt')
    del optimizer, model
    if a.device.startswith('cuda'):
        torch.cuda.empty_cache()
    model,_ = restore(checkpoint_dir/'best.pt',a.device)
    with torch.no_grad():
        restored,_ = forward_batch(model,batch,a.device)
    torch.testing.assert_close(reference,restored)
    report = dict(test_type='pretrained_software_validation_only',model=a.model,
                  model_type=model.bert.config.model_type,device=a.device,
                  pretrained_parameters=sum(p.numel() for p in model.bert.parameters()),
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  loss=float(loss.detach()),finite_gradients=gradients,
                  checkpoint_reload_matches=True,optimizer_step_completed=True,route=a.route,
                  frozen_language_model=True,patches=batch['input_ids'].shape[1],
                  diagnostic_accuracy_measured=False)
    output = Path(a.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
