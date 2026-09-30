"""Atomic checkpoints and RNG state for reproducible epoch-boundary resume."""
import random
from pathlib import Path
import numpy as np
import torch


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def rng_state():
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=(state[0],state[1].tolist(),*state[2:]),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),*n[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
