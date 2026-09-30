"""Four Table I leave-one-domain-out experiments. Launch from repository root."""
import argparse
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--data', required=True)
p.add_argument('--config', default='configs/paper.json')
p.add_argument('--output', default='runs/protocols')
p.add_argument('--fraction', type=float, default=.01)
p.add_argument('--seeds', type=int, nargs='+', default=[42,43,44])
p.add_argument('--device',default='cuda')
a = p.parse_args()
subprocess.run([sys.executable,'-m','t2mfdf.experiments','--data',a.data,
                '--config',a.config,'--output',a.output,'--fractions',str(a.fraction),
                '--device',a.device,'--seeds',*map(str,a.seeds)],check=True)
