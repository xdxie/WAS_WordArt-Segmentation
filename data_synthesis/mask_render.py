import os
import random
import numpy as np
import cv2
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


def shrink_edge_perspective(text_np, scale_range=(0.5, 0.7)):
    h, w = text_np.shape[:2]
    original_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    edge = random.choice(["top", "bottom", "left", "right"])
    scale_factor = np.random.uniform(*scale_range)
    target_pts = np.array(original_pts, dtype=np.float32)

    if edge == "top":
        shift = (1 - scale_factor) * h / 2
        target_pts[0, 1] = shift
        target_pts[1, 1] = shift
    elif edge == "bottom":
        shift = (1 - scale_factor) * h / 2
        target_pts[2, 1] = h - shift
        target_pts[3, 1] = h - shift
    elif edge == "left":
        shift = (1 - scale_factor) * w / 2
        target_pts[0, 0] = shift
        target_pts[3, 0] = shift
    else:
        shift = (1 - scale_factor) * w / 2
        target_pts[1, 0] = w - shift
        target_pts[2, 0] = w - shift

    M = cv2.getPerspectiveTransform(original_pts, target_pts)
    warped = cv2.warpPerspective(
        text_np, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )
    return warped


def render_text_image_custom(image_size, bboxes, rendered_txt_values, num_rows_values,
                             scale_range=(0.5, 0.7)):
    background = Image.new("RGB", image_size, "black")
    font_files = [f for f in os.listdir("fonts") if f.lower().endswith(".ttf")]
    if not font_files:
        raise RuntimeError("No .ttf font files found in fonts directory")

    for text, bbox, num_rows in zip(rendered_txt_values, bboxes, num_rows_values):
        font_path = os.path.join("fonts4mask", random.choice(font_files))
        font = ImageFont.truetype(font_path, size=64)
        if len(text) == 0:
            continue

        if 'ratio' not in bbox or bbox['ratio'] <= 1e-4:
            dummy_img = Image.new("RGB", (512, 512), "black")
            draw = ImageDraw.Draw(dummy_img)
            _, _, w, h = draw.textbbox((0, 0), text, font=font)
            ratio = w / h if h > 0 else 1.0
        else:
            ratio = bbox['ratio']

        width = int(bbox['width'] * image_size[0])
        height = int(width / ratio)
        top_left_x = int(bbox['top_left_x'] * image_size[0])
        top_left_y = int(bbox['top_left_y'] * image_size[1])
        yaw = bbox.get('yaw', 0)

        text_img = Image.new("RGB", (512, 512), "black")
        draw = ImageDraw.Draw(text_img)
        draw.text((10, 10), text, "white", font=font)

        text_img = resize_and_pad_image(text_img, (width, height))
        text_img_rotated = text_img.rotate(angle=-yaw, expand=True, fillcolor="black")
        text_np = np.array(text_img_rotated)

        warped = shrink_edge_perspective(text_np, scale_range=scale_range)
        warped_pil = Image.fromarray(warped)

        gray_mask = warped_pil.convert("L").point(lambda x: 255 if x > 10 else 0)
        background.paste(warped_pil, (top_left_x, top_left_y), gray_mask)

    return background


if __name__ == "__main__":
    # Parameters:
    # image_size: background canvas size (width, height)
    # bboxes: list of dicts, each controls text placement and transformation:
    #   - width: text width relative to canvas width (0~1)
    #   - top_left_x: top-left X position (0~1)
    #   - top_left_y: top-left Y position (0~1)
    #   - ratio: width/height ratio (optional, auto-computed if missing)
    #   - yaw: rotation angle in degrees
    # rendered_txt_values: list of text strings
    # num_rows_values: list of row counts (keep 1 if single-line text)
    # scale_range: range of shrinking factor, e.g. (0.5, 0.7) means 50%~70%

    image_size = (512, 512)
    bboxes = [
        {"width": 0.5, "top_left_x": 0.2, "top_left_y": 0.3, "ratio": 0.5, "yaw": 10}
    ]
    rendered_txt_values = ["Hello Mask"]
    num_rows_values = [1]
    scale_range = (0.5, 0.7)

    out_img = render_text_image_custom(
        image_size, bboxes, rendered_txt_values, num_rows_values, scale_range=scale_range
    )
    os.makedirs("masks", exist_ok=True)
    out_img.save("masks/mask_output.png")
