import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def resize_and_pad_image(img, target_size):
    old_size = img.size
    ratio = min(target_size[0] / old_size[0], target_size[1] / old_size[1])
    new_size = (int(old_size[0] * ratio), int(old_size[1] * ratio))
    img = img.resize(new_size, Image.LANCZOS)

    new_img = Image.new("RGB", target_size, "black")
    paste_x = (target_size[0] - new_size[0]) // 2
    paste_y = (target_size[1] - new_size[1]) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img


def render_text_with_font(font_path, text, image_size=(512, 512)):
    font = ImageFont.truetype(font_path, size=64)
    text_img = Image.new("RGB", (512, 512), "black")
    draw = ImageDraw.Draw(text_img)
    draw.text((10, 10), text, "white", font=font)

    text_img = resize_and_pad_image(text_img, image_size)
    gray_mask = text_img.convert("L").point(lambda x: 255 if x > 10 else 0)
    return text_img, gray_mask


if __name__ == "__main__":
    # Parameters
    text = "Font Test"  # the string to render
    image_size = (512, 512)  # canvas size
    fonts_dir = "fonts"  # folder with fonts
    out_dir = "masks"    # output folder

    os.makedirs(out_dir, exist_ok=True)
    font_files = [f for f in os.listdir(fonts_dir) if f.lower().endswith(".ttf")]

    if not font_files:
        raise RuntimeError("No .ttf font files found in fonts directory")

    for font_file in font_files:
        font_path = os.path.join(fonts_dir, font_file)
        rendered, mask = render_text_with_font(font_path, text, image_size=image_size)
        out_path = os.path.join(out_dir, f"{os.path.splitext(font_file)[0]}_mask.png")
        mask.save(out_path)
        print(f"Generated mask for {font_file} -> {out_path}")
