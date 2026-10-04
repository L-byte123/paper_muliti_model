import csv
import json
import numpy as np
from t2mfdf.prepare import prepare


def test_resampling_suppresses_alias_and_records_source(tmp_path):
    fs = 48000
    t = np.arange(fs) / fs
    # 8 kHz would alias to 4 kHz under unfiltered decimation to 12 kHz.
    signal = np.sin(2*np.pi*1000*t) + np.sin(2*np.pi*8000*t)
    np.save(tmp_path/'signal.npy', signal)
    manifest = tmp_path/'manifest.csv'
    with manifest.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['path','key','label','domain','fs','group'])
        writer.writeheader()
        writer.writerow(dict(path='signal.npy', key='', label=0, domain='CWRU', fs=fs, group='record'))
    output = tmp_path/'data.npz'
    prepare(manifest, output, snr=None, target_fs=12000)
    with np.load(output) as data:
        assert data['x'].shape == (11, 1024)
        assert np.all(data['fs'] == 12000)
        x = data['x'].reshape(-1)[100:-100]
        times = np.arange(len(x))/12000
        amplitude = lambda hz: abs(np.sum(x*np.exp(-2j*np.pi*hz*times)))
        assert amplitude(4000) < .02*amplitude(1000)
    record = json.loads((tmp_path/'data.npz.json').read_text())['records'][0]
    assert record['original_fs'] == 48000
    assert record['original_samples'] == 48000
    assert record['fs'] == 12000
    assert record['samples'] == 12000
