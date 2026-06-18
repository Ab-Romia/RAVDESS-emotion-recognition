"""Models for RAVDESS speech emotion recognition.

`SSLAudioEncoder` wraps a self-supervised speech encoder (WavLM / HuBERT /
wav2vec2) and turns a waveform into a fixed vector using two ideas that matter
for emotion: a learnable weighted sum over all transformer layers (emotion lives
in the mid and upper layers, not just the last one), and attentive statistics
pooling (the weighted mean and standard deviation over time, because prosodic
variance carries affect that plain mean pooling averages away). The same encoder
is reused by the audio-only classifier and by the multimodal model, so comparing
them is a clean ablation of what the face adds.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from .config import Config, NUM_EMOTIONS


class SSLAudioEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = AutoModel.from_pretrained(cfg.ssl_name)
        if cfg.freeze_feature_extractor and hasattr(self.backbone, "feature_extractor"):
            self.backbone.feature_extractor._freeze_parameters()

        for i, layer in enumerate(self.backbone.encoder.layers):
            if i < cfg.num_frozen_layers:
                for p in layer.parameters():
                    p.requires_grad = False
        if cfg.grad_checkpointing and self.has_trainable_backbone():
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})

        H = self.backbone.config.hidden_size
        self.use_lw = cfg.use_layer_weighting
        if self.use_lw:
            self.layer_weights = nn.Parameter(
                torch.zeros(self.backbone.config.num_hidden_layers + 1))
        self.attn = nn.Sequential(nn.Linear(H, 128), nn.Tanh(), nn.Linear(128, 1))
        self.out_dim = 2 * H

    @property
    def layers(self):
        # A property, not an attribute, so the encoder layers are not re-registered
        # as a second submodule (which would duplicate them in the state dict).
        return self.backbone.encoder.layers

    def has_trainable_backbone(self) -> bool:
        return self.cfg.num_frozen_layers < len(self.layers)

    def forward(self, input_values, attention_mask=None):
        # When the backbone is fully frozen (the linear-probe setting) run it under
        # no_grad: we never backprop into it, so there is no reason to build the graph
        # or store its activations. The head still trains, treating features as fixed.
        if self.has_trainable_backbone():
            out = self.backbone(input_values, attention_mask=attention_mask,
                                output_hidden_states=self.use_lw)
        else:
            with torch.no_grad():
                out = self.backbone(input_values, attention_mask=attention_mask,
                                    output_hidden_states=self.use_lw)
        if self.use_lw:
            stack = torch.stack(out.hidden_states, dim=0)  # (L+1, B, T, H)
            w = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
            hidden = (w * stack).sum(0)
        else:
            hidden = out.last_hidden_state

        if attention_mask is not None:
            fmask = self.backbone._get_feature_vector_attention_mask(
                hidden.shape[1], attention_mask)
        else:
            fmask = torch.ones(hidden.shape[:2], device=hidden.device)
        return self._attentive_stats(hidden, fmask)

    def _attentive_stats(self, hidden, fmask):
        scores = self.attn(hidden).squeeze(-1)
        scores = scores.masked_fill(fmask == 0, float("-inf"))
        w = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        mean = (w * hidden).sum(1)
        var = (w * (hidden - mean.unsqueeze(1)) ** 2).sum(1)
        std = torch.sqrt(var.clamp(min=1e-8))
        return torch.cat([mean, std], dim=1)

    def lr_scaled_groups(self, lr_backbone, lr_head, llrd):
        """Layer-wise LR decay over the backbone; pooling heads at the head LR."""
        groups = []
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            ps = [p for p in layer.parameters() if p.requires_grad]
            if ps:
                groups.append({"params": ps, "lr": lr_backbone * (llrd ** (n - 1 - i))})
        in_layers = {id(p) for layer in self.layers for p in layer.parameters()}
        other = [p for p in self.backbone.parameters()
                 if p.requires_grad and id(p) not in in_layers]
        if other:
            groups.append({"params": other, "lr": lr_backbone})
        head = ([self.layer_weights] if self.use_lw else []) + list(self.attn.parameters())
        groups.append({"params": head, "lr": lr_head})
        return groups


class SSLEmotionClassifier(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = SSLAudioEncoder(cfg)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.out_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, NUM_EMOTIONS),
        )

    def forward(self, input_values, attention_mask=None):
        return self.classifier(self.encoder(input_values, attention_mask))

    def param_groups(self):
        return (self.encoder.lr_scaled_groups(
                    self.cfg.lr_backbone, self.cfg.lr_head, self.cfg.llrd)
                + [{"params": self.classifier.parameters(), "lr": self.cfg.lr_head}])


class SpectrogramCNN(nn.Module):
    """From-scratch CNN over a log-magnitude STFT. No pretraining, for comparison."""

    def __init__(self, cfg: Config, n_fft: int = 512, hop: int = 160):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.register_buffer("window", torch.hann_window(n_fft))
        ch = [1, 32, 64, 128, 256]
        blocks = []
        for i in range(len(ch) - 1):
            blocks += [
                nn.Conv2d(ch[i], ch[i + 1], 3, padding=1),
                nn.BatchNorm2d(ch[i + 1]),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout), nn.Linear(ch[-1], NUM_EMOTIONS))

    def forward(self, input_values, attention_mask=None):
        spec = torch.stft(input_values, n_fft=self.n_fft, hop_length=self.hop,
                          window=self.window, return_complex=True).abs()
        spec = torch.log1p(spec).unsqueeze(1)
        x = self.features(spec)
        x = self.pool(x).flatten(1)
        return self.classifier(x)

    def param_groups(self):
        return [{"params": self.parameters(), "lr": self.cfg.lr_head}]
