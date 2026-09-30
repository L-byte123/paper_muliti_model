import csv
import json
import numpy as np
import pytest
import torch
from scipy.io import savemat
from t2mfdf.rawio import read_signal
from t2mfdf.prepare import prepare
from tests.fixtures import synthetic_fixture as demo
from t2mfdf.config import load_config
from t2mfdf.model import T2MFDF
from t2mfdf.baselines import build_model
from t2mfdf.run import train, restore
from t2mfdf.inference import predict_record
from t2mfdf.data import load_data, split_windows


def test_nested_mat_and_csv(tmp_path):
    values = np.arange(2048,dtype=np.float32)
    mat = tmp_path/'nested.mat'
    savemat(mat, {'record': {'Y': {'Data':values}}})
    np.testing.assert_array_equal(read_signal(mat,'record/Y/Data'),values)
    path = tmp_path/'channels.csv'
    np.savetxt(path,np.stack([values,-values],1),delimiter=',',header='a,b',comments='')
    np.testing.assert_array_equal(read_signal(path,column=1,skiprows=1),-values)
    with pytest.raises(ValueError):
        read_signal(path,skiprows=1)
    manifest = tmp_path/'manifest.csv'
    manifest.write_text('path,key,label,domain,fs,group\nnested.mat,record/Y/Data,1,PU,64000,PU-bearing1\n')
    prepare(manifest,tmp_path/'data.npz',snr=None)
    data = load_data(tmp_path/'data.npz')
    assert data['x'].shape == (2,1024)
    np.testing.assert_array_equal(data['x'].ravel(),values)


def test_hdf5(tmp_path):
    h5py = pytest.importorskip('h5py')
    path = tmp_path/'signal.h5'
    with h5py.File(path,'w') as f:
        f.create_dataset('measurements/channel',data=np.arange(1024))
    np.testing.assert_array_equal(read_signal(path,'measurements/channel'),np.arange(1024))


@pytest.mark.parametrize('name',['wdcnn','dlinear'])
def test_baselines(name):
    model = build_model(load_config(model_name=name,modality='time'))
    out = model(torch.randn(3,1024))
    assert out.shape == (3,4)
    out.square().mean().backward()
    assert all(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize('route',['prose','figure'])
def test_gpt_context_gradient(route):
    torch.set_num_threads(2)
    c = load_config(text_backend='tiny',backbone='gpt2',route=route,
                    d_model=32,heads=4,source_tokens=8,text_context_ff=64)
    m = T2MFDF(c).train()
    ids = torch.randint(1,128,(2,3,20))
    out = m(torch.randn(2,1024),ids,torch.ones_like(ids))
    out.sum().backward()
    assert m.text_context.layers[0].self_attn.in_proj_weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in m.bert.parameters())
    assert m.temporal.frequency.weight.grad.abs().sum() > 0


def test_resume_equivalence_and_predict(tmp_path):
    torch.set_num_threads(2)
    data = tmp_path/'data.npz'
    demo(data)
    c = load_config(modality='time',d_model=16,heads=2,batch_size=13,epochs=2,
                    patience=5,accumulation_steps=3,text_context_ff=32)
    train(data,tmp_path/'continuous',c,'cpu')
    train(data,tmp_path/'resumed',dict(c,epochs=1),'cpu')
    train(data,tmp_path/'resumed',c,'cpu',resume=True)
    full = torch.load(tmp_path/'continuous/last.pt',weights_only=True)
    resumed = torch.load(tmp_path/'resumed/last.pt',weights_only=True)
    for key in full['state_dict']:
        torch.testing.assert_close(full['state_dict'][key],resumed['state_dict'][key],rtol=0,atol=0)
    assert full['history'] == resumed['history']
    model,_ = restore(tmp_path/'resumed/best.pt','cpu')
    assert not model.training
    out = predict_record(tmp_path/'resumed/best.pt',np.arange(2500,dtype=np.float32),
                         12000,tmp_path/'pred.csv',stride=512)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 3
    assert sum(float(rows[0][f'p_class_{i}']) for i in range(4)) == pytest.approx(1)
    with pytest.raises(ValueError,match='cannot change'):
        train(data,tmp_path/'resumed',dict(c,lr=.3),'cpu',resume=True)


def test_configuration_and_window_split(tmp_path):
    with pytest.raises(ValueError,match='Unknown'):
        load_config(lern_rate=.1)
    data_path = tmp_path/'data.npz'
    demo(data_path)
    data = load_data(data_path)
    a,b,c = split_windows(data)
    assert len(set(a)&set(b)) == len(set(a)&set(c)) == 0
    assert len(set(np.concatenate([a,b,c]))) == len(data['y'])
