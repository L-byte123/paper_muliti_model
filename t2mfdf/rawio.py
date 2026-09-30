"""Explicit numeric channel selection for MATLAB, HDF5 and delimited records.

Selectors are slash separated, e.g. recording/Y/6/Data (zero-based indices).
No filename-based fault labels or silently selected channels are used.
"""
from pathlib import Path
import numpy as np


def select(value, key):
    for component in filter(None, key.split('/')):
        if isinstance(value, dict):
            value = value[component]
        elif isinstance(value, np.ndarray) and value.dtype.names:
            value = value[component]
        elif hasattr(value, component) and not component.isdecimal():
            value = getattr(value, component)
        else:
            value = value[int(component)]
    return value


def _hdf5_value(handle, value):
    import h5py
    if isinstance(value, h5py.Group):
        return {k: _hdf5_value(handle, v) for k, v in value.items() if k != '#refs#'}
    array = value[()]
    if h5py.check_dtype(ref=value.dtype) is not None:
        items = [_hdf5_value(handle, handle[r]) for r in array.ravel()]
        return items[0] if len(items) == 1 else items
    # MATLAB v7.3 stores arrays with reversed dimensions.
    if 'MATLAB_class' in value.attrs and array.ndim > 1:
        array = array.T
    return array.squeeze()


def read_signal(path, key='', column=None, skiprows=0, delimiter=None):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in ('.h5', '.hdf5', '.mat'):
        try:
            if suffix == '.mat':
                from scipy.io import loadmat
                value = select(loadmat(path, simplify_cells=True), key)
            else:
                raise NotImplementedError
        except (NotImplementedError, ValueError) as error:
            try:
                import h5py
            except ImportError as exc:
                raise ImportError('Install h5py to read HDF5/MAT v7.3 files') from exc
            if not h5py.is_hdf5(path):
                raise error
            with h5py.File(path, 'r') as handle:
                # Native dataset paths are cheap; MATLAB structs require dereferencing.
                if key in handle and isinstance(handle[key], h5py.Dataset):
                    value = _hdf5_value(handle, handle[key])
                else:
                    value = select(_hdf5_value(handle, handle), key)
    elif suffix == '.npy':
        value = np.load(path, allow_pickle=False)
    elif suffix == '.npz':
        with np.load(path, allow_pickle=False) as archive:
            value = archive[key]
    elif suffix in ('.csv', '.txt', '.dat'):
        value = np.loadtxt(path, delimiter=delimiter or (',' if suffix == '.csv' else None),
                           skiprows=int(skiprows))
    else:
        raise ValueError(f'Unsupported record format: {suffix}')
    value = np.asarray(value)
    if column is not None:
        if value.ndim != 2:
            raise ValueError(f'{path}: column requires a 2D array')
        value = value[:, int(column)]
    value = value.squeeze()
    if value.ndim != 1 or value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
        raise ValueError(f'{path}: selector must resolve to one finite real numeric channel')
    return value.astype(np.float32)
