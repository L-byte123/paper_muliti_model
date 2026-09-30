import numpy as np
import pytest
import torch
from t2mfdf.features import describe, add_noise, patch_texts
from t2mfdf.data import load_data, split_groups, SignalDataset
from tests.fixtures import synthetic_fixture as demo
from t2mfdf.model import T2MFDF
from t2mfdf.run import DEFAULT, metrics, balance_indices


def test_features_and_noise():
    x = np.sin(2*np.pi*8*np.arange(128)/128)
    f = describe(x, 128)
    assert f['dominant_bin'] == 8
    assert f['dominant_hz'] == 8
    assert f['spectral_concentration'] == pytest.approx(1.)
    assert describe(np.zeros(32), 12000)['wavelet_entropy'] == 0
    assert len(patch_texts(np.zeros(1024), 12000)) == 63
    noisy = add_noise(x[None], 10, np.random.default_rng(1))
    measured = 10*np.log10(np.mean(x*x)/np.mean((noisy-x)**2))
    assert measured == pytest.approx(10, abs=1e-5)


def test_group_integrity(tmp_path):
    path = tmp_path/'demo.npz'
    demo(path)
    data = load_data(path)
    splits = split_groups(data)
    for i in range(3):
        for j in range(i):
            assert not set(data['groups'][splits[i]]) & set(data['groups'][splits[j]])
    assert len(set(np.concatenate(splits))) == len(data['x'])


@pytest.mark.parametrize('modality,route', [('time','prose'), ('text','prose'),
                                         ('multimodal','prose'), ('multimodal','figure')])
def test_gradients_and_reload(modality, route):
    torch.set_num_threads(2)
    c = dict(DEFAULT, d_model=32, heads=4, text_backend='tiny', modality=modality,
             route=route, source_tokens=8)
    model = T2MFDF(c).train()
    x = torch.randn(2,1024)
    ids = torch.randint(1,128,(2,3,16))
    mask = torch.ones_like(ids)
    out = model(x,ids,mask)
    assert out.shape == (2,4)
    torch.nn.functional.cross_entropy(out,torch.tensor([0,1])).backward()
    assert model.classifier.weight.grad.abs().sum() > 0
    if modality != 'text':
        assert model.temporal.lstm.weight_ih_l0.grad.abs().sum() > 0
    if model.bert is not None:
        assert all(p.grad is None for p in model.bert.parameters())
        assert not model.bert.training
    if route == 'figure':
        assert model.vocabulary_mapping.weight.grad.abs().sum() > 0
        assert model.to_bert.weight.grad.abs().sum() > 0
    model.eval()
    clone = T2MFDF(model.config, load_pretrained=False).eval()
    clone.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(x,ids,mask), clone(x,ids,mask))


def test_metrics():
    m = metrics([0,0,1,1], [0,1,1,1], 2)
    assert m['accuracy'] == .75
    assert m['macro_f1'] == pytest.approx((2/3+.8)/2)
    assert m['confusion_matrix'] == [[1,1],[0,2]]


def test_balance_ratios():
    data = dict(y=np.repeat(np.arange(4), 12), domains=np.repeat('CWRU', 48))
    selected = balance_indices(data, np.arange(48), 42)
    assert np.bincount(data['y'][selected]).tolist() == [4,12,12,12]
    assert len(set(selected)) == len(selected)
