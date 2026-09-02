"""
Single-image LGD inference — no dataset required.
"""
import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from hardware.device import get_device
from inference.post_process import post_process_output
from utils.model_util import create_diffusion
from utils.dataset_processing import grasp as grasp_utils


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image', type=str, required=True)
    p.add_argument('--prompt', type=str, required=True)
    p.add_argument('--network', type=str, default='checkpoints/lgd_pretrained.pth')
    p.add_argument('--input-size', type=int, default=224)
    p.add_argument('--cpu', dest='force_cpu', action='store_true', default=True)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = get_device(args.force_cpu)

    img = Image.open(args.image).convert('RGB')
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    img = img.resize((args.input_size, args.input_size))
    rgb = np.array(img).astype(np.float32) / 255.0
    rgb = rgb.transpose((2, 0, 1))
    rgb -= rgb.mean()
    x = torch.from_numpy(rgb).unsqueeze(0).float().to(device)

    print('Loading network...')
    net = torch.load(args.network, map_location=device)
    net.eval()

    diffusion = create_diffusion()

    pos_shape = (1, 1, args.input_size, args.input_size)
    gt = torch.zeros(pos_shape, device=device)
    alpha = 0.4
    idx = torch.zeros(1, device=device)

    print('Sampling (slow on CPU)...')
    with torch.no_grad():
        sample = diffusion.p_sample_loop(net, pos_shape, gt, x, [args.prompt], alpha, idx)
        pos_output = sample
        cos_output = net.cos_output_str
        sin_output = net.sin_output_str
        width_output = net.width_output_str
        q_img, ang_img, width_img = post_process_output(pos_output, cos_output, sin_output, width_output)

    gs = grasp_utils.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.array(img))
    for g in gs:
        g.plot(ax)
    ax.set_title(args.prompt)
    ax.axis('off')
    import re
    safe_prompt = re.sub(r'[^a-zA-Z0-9]+', '_', args.prompt)[:40]
    outname = f'result_{safe_prompt}.png'
    fig.savefig(outname, bbox_inches='tight')
    print(f'Saved {outname}')
