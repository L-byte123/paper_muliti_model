import numpy as np
import torch
from t2mfdf.config import load_config
from t2mfdf.run import train, restore, loader, forward_batch


def test_multimodal_full_training(tmp_path):
    torch.set_num_threads(2)
    rng = np.random.default_rng(12)
    data = dict(x=rng.normal(size=(12,1024)).astype(np.float32),
                y=np.repeat(np.arange(4),3),fs=np.full(12,12000),
                groups=np.array([str(i) for i in range(12)]),domains=np.repeat('CWRU',12))
    path = tmp_path/'fixture.npz'
    np.savez(path,**data)
    config = load_config(text_backend='tiny',d_model=16,heads=2,batch_size=2,
        epochs=1,source_tokens=8,bert_chunk_size=128,max_text_length=24,text_context_ff=32)
    report = train(path,tmp_path/'run',config,'cpu')
    assert report['counts'] == dict(train=4,val=4,test=4,target=0)
    assert report['smoke_test_only']
    restored,tokenizer = restore(tmp_path/'run/best.pt','cpu')
    out,_ = forward_batch(restored,next(iter(loader(data,np.array([0]),config,tokenizer))),'cpu')
    assert torch.isfinite(out).all()
