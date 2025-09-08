import os
import cv2
import json
import random
import torch
import einops
import argparse
import numpy as np
from tqdm import tqdm
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


def process(image_path, prompt, a_prompt, n_prompt, num_samples, image_resolution,
            ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold,
            model, ddim_sampler):
    """
    Run ControlNet generation on one image.
    """
    with torch.no_grad():
        img = cv2.imread(image_path)
        img = resize_image(HWC3(img), image_resolution)
        H, W, C = img.shape

        control = torch.from_numpy(img.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        cond = {
            "c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]
        }
        shape = (4, H // 8, W // 8)

        model.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)]
            if guess_mode else ([strength] * 13)
        )

        samples, _ = ddim_sampler.sample(
            ddim_steps, num_samples, shape, cond, verbose=False, eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond
        )

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5)
        x_samples = x_samples.cpu().numpy().clip(0, 255).astype(np.uint8)

    return x_samples


def read_nth_line(file_path, line_number):
    """Read nth line (1-based) from file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if 1 <= line_number <= len(lines):
            return lines[line_number - 1].strip()
        else:
            raise ValueError(f"Line {line_number} is out of range in {file_path}")


def extract_prompt(input_string):
    """Extract 'prompt' field from a JSON string."""
    data = json.loads(input_string)
    return data.get("prompt", "")


def main(args):
    # Load model
    print(f"Loading model from {args.ckpt} ...")
    model = create_model(args.config)
    model.load_state_dict(load_state_dict(args.ckpt, location='cpu'), strict=False)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    os.makedirs(args.target_folder, exist_ok=True)

    for filename in tqdm(os.listdir(args.source_folder), desc="Processing files"):
        if not filename.endswith(".png"):
            continue

        target_file = os.path.join(args.target_folder, filename.replace(".png", ".jpg"))
        if os.path.exists(target_file):
            continue

        image_path = os.path.join(args.source_folder, filename)
        line_number = int(filename.split(".")[0])
        prompt_line = read_nth_line(args.prompt_file, line_number)
        prompt_text = extract_prompt(prompt_line)

        result = process(
            image_path=image_path,
            prompt=prompt_text,
            a_prompt="high contrast, best quality, extremely detailed",
            n_prompt="cropped, worst quality, low quality",
            num_samples=1,
            image_resolution=args.image_resolution,
            ddim_steps=args.ddim_steps,
            guess_mode=False,
            strength=1.0,
            scale=9.0,
            seed=42,
            eta=0.0,
            low_threshold=100,
            high_threshold=200,
            model=model,
            ddim_sampler=ddim_sampler
        )

        cv2.imwrite(target_file, result[0])

    print(f"All results saved to {args.target_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ControlNet Batch Inference Script")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="./models/cldm_v15.yaml", help="Model config file")
    parser.add_argument("--source_folder", type=str, required=True, help="Input image folder")
    parser.add_argument("--target_folder", type=str, required=True, help="Output image folder")
    parser.add_argument("--prompt_file", type=str, required=True, help="JSON file containing prompts")
    parser.add_argument("--image_resolution", type=int, default=648, help="Image resolution")
    parser.add_argument("--ddim_steps", type=int, default=15, help="DDIM steps")

    args = parser.parse_args()
    main(args)
