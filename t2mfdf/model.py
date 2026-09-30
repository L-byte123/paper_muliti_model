"""Eq. 1, Eqs. 10-16, with separately selectable Fig. 1 reprogramming route."""
import math
import torch
from torch import nn


def position_encoding(length, dim, device, dtype):
    position = torch.arange(length, device=device).unsqueeze(1)
    scale = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(position * scale), torch.cos(position * scale)
    return pe.to(dtype)


class TemporalEncoder(nn.Module):
    def __init__(self, dim, patch_length, stride):
        super().__init__()
        self.patch_length, self.stride = patch_length, stride
        self.lstm = nn.LSTM(1, dim // 2, batch_first=True)
        self.frequency = nn.Linear(patch_length, dim // 2)

    def forward(self, x):
        patches = x.unfold(-1, self.patch_length, self.stride)
        b, n, p = patches.shape
        _, (hidden, _) = self.lstm(patches.reshape(b * n, p, 1))
        # Eq. 1 specifies REAL DFT, not FFT magnitude.
        freq = self.frequency(torch.fft.fft(patches.float(), dim=-1).real.to(x.dtype))
        result = torch.cat([hidden[-1].reshape(b, n, -1), freq], dim=-1)
        return result + position_encoding(n, result.size(-1), x.device, result.dtype)


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.branch = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.branch(x)


class T2MFDF(nn.Module):
    def __init__(self, config, load_pretrained=True):
        super().__init__()
        self.config = dict(config)
        c = self.config
        dim = c['d_model']
        if dim % 2 or dim % c['heads']:
            raise ValueError("d_model must be even and divisible by heads")
        self.modality, self.route = c['modality'], c['route']
        self.temporal = TemporalEncoder(dim, c['patch_length'], c['patch_stride'])
        self.bert = None
        if self.modality != 'time':
            from transformers import BertConfig, BertModel, GPT2Config, GPT2Model
            is_gpt = c.get('backbone', 'bert') == 'gpt2'
            model_cls = GPT2Model if is_gpt else BertModel
            config_cls = GPT2Config if is_gpt else BertConfig
            kwargs = {} if is_gpt else {'add_pooling_layer': False}
            if c['text_backend'] == 'tiny':
                bc = BertConfig(vocab_size=128, hidden_size=32, num_hidden_layers=1,
                                num_attention_heads=4, intermediate_size=64,
                                max_position_embeddings=512)
                if is_gpt:
                    bc = GPT2Config(vocab_size=128, n_embd=32, n_layer=1, n_head=4, n_positions=512)
                self.bert = model_cls(bc, **kwargs)
            elif load_pretrained:
                self.bert = model_cls.from_pretrained(c['bert_model'], **kwargs)
            else:
                self.bert = model_cls(config_cls.from_dict(c['bert_config']), **kwargs)
            self.bert.config.use_cache = False
            if c['max_text_length'] > self.bert.config.max_position_embeddings:
                raise ValueError('max_text_length exceeds language model position capacity')
            self.config['bert_config'] = self.bert.config.to_dict()
            self.bert.requires_grad_(False)
            width = self.bert.config.hidden_size
            self.text_projection = nn.Linear(width, dim)
            layers = c.get('text_context_layers', 0)
            self.text_context = (nn.TransformerEncoder(nn.TransformerEncoderLayer(
                dim, c['heads'], c.get('text_context_ff', 4 * dim), c['dropout'],
                activation='gelu', batch_first=True), layers, enable_nested_tensor=False)
                if layers else None)
            if self.modality == 'multimodal' and (self.route == 'figure' or c.get('use_token_prototypes',False)):
                self.vocabulary_mapping = nn.Linear(self.bert.config.vocab_size,
                                                    c['source_tokens'], bias=False)
            if self.route == 'figure' and self.modality == 'multimodal':
                self.to_bert = nn.Linear(dim, width)
        self.q_norm, self.k_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, c['heads'], dropout=c['dropout'], batch_first=True)
        self.output_norm = nn.LayerNorm(dim)
        self.residual = nn.Sequential(ResidualBlock(dim, c['dropout']), ResidualBlock(dim, c['dropout']))
        self.classifier = nn.Linear(dim, c['num_classes'])

    def train(self, mode=True):
        super().train(mode)
        if self.bert is not None:
            self.bert.eval()  # Frozen also means deterministic dropout during training.
        return self

    def contextual_text(self, ids, mask):
        b, n, length = ids.shape
        flat_ids, flat_mask = ids.flatten(0, 1), mask.flatten(0, 1)
        chunks = []
        with torch.no_grad():
            for i in range(0, len(flat_ids), self.config['bert_chunk_size']):
                m = flat_mask[i:i + self.config['bert_chunk_size']]
                h = self.bert(input_ids=flat_ids[i:i + len(m)], attention_mask=m).last_hidden_state
                if self.config['text_scope'] == 'window':
                    chunks.append(h)
                else:  # Algorithm 1: one contextual descriptor per patch.
                    chunks.append((h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp_min(1))
        if self.config['text_scope'] == 'window':
            text = torch.cat(chunks).reshape(b, length, -1)
            padding = ~mask[:, 0].bool()
        else:
            text = torch.cat(chunks).reshape(b, n, -1)
            padding = torch.zeros(b, n, device=ids.device, dtype=torch.bool)
        text = self.text_projection(text)
        if self.text_context is not None:
            # Eq. 9: contextualize the descriptor sequence across patches.
            text = text + position_encoding(text.size(1), text.size(2), text.device, text.dtype)
            text = self.text_context(text, src_key_padding_mask=padding)
        return text, padding

    def forward(self, signal, input_ids=None, attention_mask=None):
        if self.modality == 'time':
            features = self.temporal(signal)
        else:
            text, padding = self.contextual_text(input_ids, attention_mask)
            if self.modality == 'text':
                text = self.residual(text)
                return self.classifier((text * (~padding).unsqueeze(-1)).sum(1) /
                                       (~padding).sum(1, keepdim=True).clamp_min(1))
            temporal = self.temporal(signal)
            if hasattr(self, 'vocabulary_mapping'):
                vocab = self.bert.get_input_embeddings().weight.detach()
                source = self.vocabulary_mapping(vocab.T).T
                source = self.text_projection(source).unsqueeze(0).expand(signal.size(0), -1, -1)
                text = torch.cat([text, source], dim=1)
                padding = torch.cat([padding, torch.zeros(source.shape[:2], device=signal.device,
                                                         dtype=torch.bool)], dim=1)
            q, k = self.q_norm(temporal), self.k_norm(text)
            features = self.output_norm(self.attention(q, k, text, key_padding_mask=padding,
                                                       need_weights=False)[0])
            if self.route == 'figure':
                # Do NOT use no_grad: gradients must flow through frozen BERT to alignment.
                features = self.text_projection(self.bert(inputs_embeds=self.to_bert(features),
                    attention_mask=torch.ones(features.shape[:2], device=signal.device)).last_hidden_state)
        return self.classifier(self.residual(features).mean(dim=1))
