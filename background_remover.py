#!/usr/bin/env python3
"""
Background Remover Script
Author: Pablo R (murapadev)
Description: Removes backgrounds from images in the input folder using a Hugging Face model.
"""

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageChops

# Native transformers architecture (no remote code), MIT licensed.
DEFAULT_MODEL = "tue-mps/coco_instance_eomt_large_640"
DEFAULT_REVISION = "a43666870d47a7010bfea2c4554f7ca829714558"


def _load_pipeline(model_name: str, revision: str, trust_remote_code: bool = False):
    from transformers import pipeline

    return pipeline(
        "image-segmentation",
        model=model_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )


def _union_masks(masks: list[Image.Image], size: tuple[int, int]) -> Image.Image:
    alpha = Image.new("L", size, 0)
    for mask in masks:
        mask = mask.convert("L")
        if mask.size != size:
            mask = mask.resize(size, Image.Resampling.BILINEAR)
        alpha = ImageChops.lighter(alpha, mask)
    return alpha


def to_rgba(image: Image.Image, result: object) -> Image.Image:
    """Build an RGBA cutout from whatever the segmentation pipeline returned.

    - PIL RGBA/LA image: custom pipelines (e.g. RMBG-1.4) that already return the cutout.
    - PIL L/1/I/F image: a foreground mask.
    - list of {"label", "score", "mask"} dicts: the stock transformers pipeline;
      the foreground is the union of all returned masks.
    """
    rgb = image.convert("RGB")

    if isinstance(result, dict):
        result = [result]

    if isinstance(result, Image.Image):
        if result.mode in ("RGBA", "LA"):
            alpha = result.getchannel("A")
        elif result.mode in ("L", "1", "I", "F"):
            alpha = result.convert("L")
        else:
            raise TypeError(f"Cannot use a {result.mode} image as a mask")
        masks = [alpha]
    elif isinstance(result, list):
        masks = [item["mask"] for item in result if isinstance(item, dict) and "mask" in item]
        if not masks:
            raise ValueError("No foreground detected")
        if not all(isinstance(m, Image.Image) for m in masks):
            raise TypeError("Pipeline returned masks that are not PIL images")
    else:
        raise TypeError(f"Unexpected pipeline output type: {type(result).__name__}")

    rgba = rgb.copy()
    rgba.putalpha(_union_masks(masks, rgb.size))
    return rgba


def remove_background(
    input_path: str,
    output_path: str,
    model_name: str = DEFAULT_MODEL,
    revision: str = DEFAULT_REVISION,
    trust_remote_code: bool = False,
) -> int:
    """Remove background from images. Returns 0 on success, 1 on partial failure, -1 on no input."""
    input_dir = Path(input_path)
    output_dir = Path(output_path)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: input directory '{input_dir}' does not exist or is not a directory.")
        return -1

    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = _load_pipeline(model_name, revision, trust_remote_code)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    success = 0
    total = 0

    for image_file in sorted(input_dir.rglob("*")):
        if not image_file.is_file():
            continue
        if image_file.suffix.lower() not in image_extensions:
            continue

        total += 1
        print(f"Processing {image_file.name}...")
        try:
            image = Image.open(image_file).convert("RGB")
            result_image = to_rgba(image, pipe(image))

            relative_path = image_file.relative_to(input_dir)
            output_file = output_dir / relative_path.with_name(
                f"{relative_path.stem}_no_bg.png"
            )
            output_file.parent.mkdir(parents=True, exist_ok=True)
            result_image.save(output_file, format="PNG")
            print(f"  -> {output_file}")
            success += 1
        except Exception as e:
            print(f"  Error processing {image_file.name}: {e}")

    if total == 0:
        print(f"No supported images found in '{input_dir}'.")
        return -1

    print(f"\nDone: {success}/{total} images processed successfully.")
    return 0 if success == total else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove backgrounds from images.")
    parser.add_argument("--input", default="./input", help="Input folder with images")
    parser.add_argument("--output", default="./output", help="Output folder for processed images")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Model revision; pin a commit SHA for reproducibility.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow models that ship their own Python code (e.g. briaai/RMBG-1.4). "
        "That code runs on your machine: only enable it for repos you trust.",
    )
    args = parser.parse_args()
    sys.exit(
        remove_background(
            args.input, args.output, args.model, args.revision, args.trust_remote_code
        )
    )
