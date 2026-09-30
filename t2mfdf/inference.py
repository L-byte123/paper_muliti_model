"""Batch diagnosis of continuous recordings using a trained, offline checkpoint."""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from .rawio import read_signal
from .run import restore, loader, forward_batch


def predict_record(checkpoint, signal, fs, output, stride=1024, device='cpu'):
    if stride < 1 or len(signal) < 1024 or not np.isfinite(fs) or fs <= 0:
        raise ValueError('Need >=1024 samples, positive stride and finite positive sample rate')
    model, tokenizer = restore(checkpoint, device)
    starts = np.arange(0,len(signal)-1024+1,stride)
    windows = np.stack([signal[i:i+1024] for i in starts]).astype(np.float32)
    if not np.isfinite(windows).all():
        raise ValueError('Nonfinite signal')
    data = dict(x=windows,y=np.zeros(len(windows),dtype=np.int64),fs=np.full(len(windows),fs))
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',encoding='utf8',newline='') as f, torch.no_grad():
        writer = csv.writer(f)
        writer.writerow(['start_sample','end_sample','start_seconds','label']+
                        [f'p_class_{i}' for i in range(model.config['num_classes'])])
        offset = 0
        for batch in loader(data,np.arange(len(windows)),model.config,tokenizer):
            logits,_ = forward_batch(model,batch,device)
            for probabilities in logits.softmax(-1).cpu().numpy():
                start = int(starts[offset])
                writer.writerow([start,start+1024,start/fs,int(probabilities.argmax()),*probabilities.tolist()])
                offset += 1
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--input',required=True)
    p.add_argument('--key',default='')
    p.add_argument('--column',type=int)
    p.add_argument('--skiprows',type=int,default=0)
    p.add_argument('--fs',required=True,type=float)
    p.add_argument('--stride',type=int,default=1024)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu')
    a = p.parse_args()
    signal = read_signal(a.input,a.key,a.column,a.skiprows)
    print(predict_record(a.checkpoint,signal,a.fs,a.output,a.stride,a.device))


if __name__ == '__main__':
    main()
