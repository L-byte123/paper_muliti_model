"""Explicit classification adaptations; these are not the authors' baseline code."""
import torch
from torch import nn
from torch.nn import functional as F


class WDCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        blocks = []
        for cin, cout, kernel, stride, padding in [(1,16,64,16,24),(16,32,3,1,1),
                (32,64,3,1,1),(64,64,3,1,1),(64,64,3,1,1)]:
            blocks.extend([nn.Conv1d(cin,cout,kernel,stride,padding), nn.BatchNorm1d(cout),
                           nn.ReLU(), nn.MaxPool1d(2)])
        self.encoder = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128,100), nn.ReLU(),
            nn.Dropout(config['dropout']), nn.Linear(100,config['num_classes']))

    def forward(self, signal, **kwargs):
        return self.classifier(self.encoder(signal.unsqueeze(1)))


class DLinear(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.trend = nn.Linear(1024,config['num_classes'])
        self.seasonal = nn.Linear(1024,config['num_classes'])

    def forward(self, signal, **kwargs):
        trend = F.avg_pool1d(F.pad(signal.unsqueeze(1), (12,12), mode='replicate'),
                             25, stride=1).squeeze(1)
        return self.trend(trend) + self.seasonal(signal-trend)


def build_model(config, load_pretrained=True):
    from .model import T2MFDF
    name = config.get('model_name', 't2mfdf')
    if name == 't2mfdf':
        return T2MFDF(config, load_pretrained)
    return {'wdcnn': WDCNN, 'dlinear': DLinear}[name](config)
