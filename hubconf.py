"""
Allows to load pre-trained models from the `ibot` repository.

Example:
```python
import torch
import torch.hub

model = torch.hub.load("bytedance/ibot", "vits_16")
"""

import torch
import torch.hub
import torch.nn as nn

import models.swin_transformer as st
import models.vision_transformer as vt

URL = "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/"

PTHS = dict(
    vit_s16="vits_16/checkpoint_teacher.pth",
    swint_7="swint_7/checkpoint_teacher.pth",
    swint_14="swint_14/checkpoint_teacher.pth",
    vitb_16="vitb_16/checkpoint_teacher.pth",
    vitb_16_rand_mask="vitb_16_rand_mask/checkpoint_teacher.pth",
    vitl_16="vitl_16/checkpoint_teacher.pth",
    vitl_16_rand_mask="vitl_16_rand_mask/checkpoint_teacher.pth",
)


def _load_ckpt(pth, model: nn.Module, pretrained=True, **kwargs):
    if pretrained:
        pth = torch.hub.load_state_dict_from_url(
            url=URL + pth, file_name="ibot_" + pth.split("/")[0]
        )
        state_dict = pth["state_dict"]
        model.load_state_dict(state_dict, strict=False)
    return model


def vits_16(**kwargs):
    model = vt.vit_small(**kwargs)
    return _load_ckpt(PTHS["vit_s16"], model)


def swint_7(**kwargs):
    model = st.swin_tiny(**kwargs)
    return _load_ckpt(PTHS["swint_7"], model)


def swint_14(**kwargs):
    model = st.swin_tiny(**kwargs, window_size=14)
    return _load_ckpt(PTHS["swint_14"], model)


def vitb_16(**kwargs):
    model = vt.vit_base(**kwargs)
    return _load_ckpt(PTHS["vitb_16"], model)


def vitb_16_rand_mask(**kwargs):
    model = vt.vit_base(**kwargs)
    return _load_ckpt(PTHS["vitb_16_rand_mask"], model)


def vitl_16(**kwargs):
    model = vt.vit_large(**kwargs)
    return _load_ckpt(PTHS["vitl_16"], model)


def vitl_16_rand_mask(**kwargs):
    model = vt.vit_large(**kwargs)
    return _load_ckpt(PTHS["vitl_16_rand_mask"], model)


dependencies = ["torch"]
