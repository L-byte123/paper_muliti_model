"""Controlled modality ablations, using the same seed and group splits."""
import argparse
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--data', required=True)
p.add_argument('--config', default='configs/paper.json')
p.add_argument('--output', default='runs/ablations')
p.add_argument('--device', default='cuda')
p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
a = p.parse_args()
subprocess.run([sys.executable,'-m','t2mfdf.experiments','--data',a.data,
                '--config',a.config,'--output',a.output,'--variants','full','time','text','gpt2',
                '--device',a.device,'--seeds',*map(str,a.seeds)],check=True)
