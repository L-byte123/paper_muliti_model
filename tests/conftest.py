"""Support a nested --basetemp on a fresh checkout without a tmp directory."""
from pathlib import Path


def pytest_configure(config):
    base = config.getoption('basetemp')
    if base:
        Path(base).resolve().parent.mkdir(parents=True, exist_ok=True)
