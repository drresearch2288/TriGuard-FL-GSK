"""Frozen vision backbones split into a cached head part (blocks 1..n_frozen) and a trainable tail."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

import triguard.paths  # noqa: F401  (sets HF_HOME / TORCH_HOME)

BACKBONES = {
    # name: (loader, mean, std, n_blocks)
    "dinov2_vits14": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "vit_s_in": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "biomedclip": ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
}


@dataclass
class Backbone:
    name: str
    model: nn.Module
    blocks: nn.ModuleList
    norm: nn.Module
    mean: tuple
    std: tuple
    kind: str  # 'dinov2' | 'timm'

    @property
    def dim(self) -> int:
        return self.norm.weight.shape[0]

    @torch.no_grad()
    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Patch embedding + positional embedding (+ cls). Input already normalised."""
        if self.kind == "dinov2":
            return self.model.prepare_tokens_with_masks(x)
        m = self.model
        x = m.patch_embed(x)
        x = m._pos_embed(x)
        x = m.patch_drop(x)
        return m.norm_pre(x)

    @torch.no_grad()
    def forward_frozen(self, x: torch.Tensor, n_frozen: int) -> torch.Tensor:
        t = self.tokens(x)
        for b in self.blocks[:n_frozen]:
            t = b(t)
        return t

    @torch.no_grad()
    def forward_full_features(self, x: torch.Tensor) -> torch.Tensor:
        t = self.tokens(x)
        for b in self.blocks:
            t = b(t)
        t = self.norm(t)
        return torch.cat([t[:, 0], t[:, 1:].mean(1)], dim=-1)


def load_backbone(name: str, device="cpu") -> Backbone:
    mean, std = BACKBONES[name]
    if name == "dinov2_vits14":
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)
        bb = Backbone(name, m, m.blocks, m.norm, mean, std, "dinov2")
    elif name == "vit_s_in":
        import timm

        m = timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=True)
        bb = Backbone(name, m, m.blocks, m.norm, mean, std, "timm")
    elif name == "biomedclip":
        import open_clip

        clip, _, _ = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        m = clip.visual.trunk
        bb = Backbone(name, m, m.blocks, m.norm, mean, std, "timm")
    else:
        raise KeyError(name)
    bb.model.eval().requires_grad_(False).to(device)
    return bb
