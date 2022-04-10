#!/usr/bin/env python

from __future__ import annotations

import argparse
import functools
import os
import subprocess
import sys

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

if os.environ.get('SYSTEM') == 'spaces':
    subprocess.call('git apply ../patch'.split(), cwd='stylegan2-pytorch')

sys.path.insert(0, 'stylegan2-pytorch')

from model import Generator

TITLE = 'TADNE (This Anime Does Not Exist)'
DESCRIPTION = '''The original TADNE site is https://thisanimedoesnotexist.ai/.
The model used here is the one converted from the model provided in [this site](https://www.gwern.net/Faces) using [this repo](https://github.com/rosinality/stylegan2-pytorch).
'''
SAMPLE_IMAGE_DIR = 'https://huggingface.co/spaces/hysts/TADNE/resolve/main/samples'
ARTICLE = f'''## Generated images
- size: 512x512
- truncation: 0.7
- seed: 0-99
![samples]({SAMPLE_IMAGE_DIR}/sample.jpg)
'''

TOKEN = os.environ['TOKEN']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    parser.add_argument('--allow-screenshot', action='store_true')
    return parser.parse_args()


def load_model(device: torch.device) -> nn.Module:
    model = Generator(512, 1024, 4, channel_multiplier=2)
    path = hf_hub_download('hysts/TADNE',
                           'models/aydao-anime-danbooru2019s-512-5268480.pt',
                           use_auth_token=TOKEN)
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


def main():
    gr.close_all()

    args = parse_args()
    device = torch.device(args.device)

    model = load_model(device)

    func = functools.partial(generate_image, model=model, device=device)
    func = functools.update_wrapper(func, generate_image)

    gr.Interface(
        func,
        [
            gr.inputs.Number(default=55376, label='Seed'),
            gr.inputs.Slider(
                0, 2, step=0.05, default=0.7, label='Truncation psi'),
            gr.inputs.Checkbox(default=False, label='Randomize Noise'),
        ],
        gr.outputs.Image(type='numpy', label='Output'),
        title=TITLE,
        description=DESCRIPTION,
        article=ARTICLE,
        theme=args.theme,
        allow_screenshot=args.allow_screenshot,
        allow_flagging=args.allow_flagging,
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
