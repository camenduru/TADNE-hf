#!/usr/bin/env python

from __future__ import annotations

import functools
import os
import random
import shlex
import subprocess
import sys

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

if os.environ.get('SYSTEM') == 'spaces':
    with open('patch') as f:
        subprocess.run(shlex.split('patch -p1'),
                       cwd='stylegan2-pytorch',
                       stdin=f)
    if not torch.cuda.is_available():
        with open('patch-cpu') as f:
            subprocess.run(shlex.split('patch -p1'),
                           cwd='stylegan2-pytorch',
                           stdin=f)

sys.path.insert(0, 'stylegan2-pytorch')

from model import Generator

DESCRIPTION = '''# [TADNE](https://thisanimedoesnotexist.ai/) (This Anime Does Not Exist)

Related Apps:
- [TADNE Image Viewer](https://huggingface.co/spaces/hysts/TADNE-image-viewer)
- [TADNE Image Selector](https://huggingface.co/spaces/hysts/TADNE-image-selector)
- [TADNE Interpolation](https://huggingface.co/spaces/hysts/TADNE-interpolation)
- [TADNE Image Search with DeepDanbooru](https://huggingface.co/spaces/hysts/TADNE-image-search-with-DeepDanbooru)
'''
SAMPLE_IMAGE_DIR = 'https://huggingface.co/spaces/hysts/TADNE/resolve/main/samples'
ARTICLE = f'''## Generated images
- size: 512x512
- truncation: 0.7
- seed: 0-99
![samples]({SAMPLE_IMAGE_DIR}/sample.jpg)
'''

MAX_SEED = np.iinfo(np.int32).max


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def load_model(device: torch.device) -> nn.Module:
    model = Generator(512, 1024, 4, channel_multiplier=2)
    path = hf_hub_download('public-data/TADNE',
                           'models/aydao-anime-danbooru2019s-512-5268480.pt')
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['g_ema'])
    model.eval()
    model.to(device)
    model.latent_avg = checkpoint['latent_avg'].to(device)
    with torch.inference_mode():
        z = torch.zeros((1, model.style_dim)).to(device)
        model([z], truncation=0.7, truncation_latent=model.latent_avg)
    return model


def generate_z(z_dim: int, seed: int, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.random.RandomState(seed).randn(
        1, z_dim)).to(device).float()


@torch.inference_mode()
def generate_image(seed: int, truncation_psi: float, randomize_noise: bool,
                   model: nn.Module, device: torch.device) -> np.ndarray:
    seed = int(np.clip(seed, 0, np.iinfo(np.uint32).max))

    z = generate_z(model.style_dim, seed, device)
    out, _ = model([z],
                   truncation=truncation_psi,
                   truncation_latent=model.latent_avg,
                   randomize_noise=randomize_noise)
    out = (out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return out[0].cpu().numpy()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = load_model(device)
fn = functools.partial(generate_image, model=model, device=device)

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            seed = gr.Slider(label='Seed',
                             minimum=0,
                             maximum=MAX_SEED,
                             step=1,
                             value=0)
            randomize_seed = gr.Checkbox(label='Randomize seed', value=True)
            psi = gr.Slider(label='Truncation psi',
                            minimum=0,
                            maximum=2,
                            step=0.05,
                            value=0.7)
            randomize_noise = gr.Checkbox(label='Randomize Noise', value=False)
            run_button = gr.Button('Run')
        with gr.Column():
            result = gr.Image(label='Output')
    gr.Markdown(ARTICLE)

    run_button.click(
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=fn,
        inputs=[seed, psi, randomize_noise],
        outputs=result,
        api_name='run',
    )
demo.queue(max_size=10).launch()
