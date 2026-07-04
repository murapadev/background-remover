#!/usr/bin/env python3
"""
Background Remover Script
Author: Pablo R (murapadev)
Description: Removes backgrounds from images in the input folder using a Hugging Face model.
"""

import argparse
import sys
from pathlib import Path
from transformers import pipeline
from PIL import Image


DEFAULT_MODEL = "briaai/RMBG-2.0"
PUBLIC_FALLBACK_MODEL = "briaai/RMBG-1.4"

# Pinneadas para evitar ejecución de código remoto arbitrario si el repo cambia.
# Actualizar periódicamente al último commit verificado del modelo.
DEFAULT_REVISION = "main"
PUBLIC_FALLBACK_REVISION = "main"


def _load_pipeline(model_name: str, revision: str) -> "pipeline":
    try:
        return pipeline(
            "image-segmentation",
            model=model_name,
            trust_remote_code=True,
            revision=revision,
        )
    except Exception as exc:
        error_text = str(exc).lower()
        is_gated = (
            "gated repo" in error_text
            or "401" in error_text
            or "403" in error_text
            or "access restricted" in error_text
        )
        if is_gated and model_name != PUBLIC_FALLBACK_MODEL:
            print(
                f"Model '{model_name}' requires Hugging Face access/token. "
                f"Falling back to public model '{PUBLIC_FALLBACK_MODEL}'."
            )
            print(
                "Tip: run `huggingface-cli login` or set HF_TOKEN to use gated models."
            )
            return pipeline(
                "image-segmentation",
                model=PUBLIC_FALLBACK_MODEL,
                trust_remote_code=True,
                revision=PUBLIC_FALLBACK_REVISION,
            )
        raise


def _extract_output_image(result: object) -> Image.Image:
    """Handle both PIL.Image (RMBG direct) and list-of-dicts (vanilla pipeline)."""
    if isinstance(result, Image.Image):
        return result
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
        mask = result[0].get("mask")
        if isinstance(mask, Image.Image):
            return mask
        raise TypeError(
            f"Expected PIL.Image in result dict, got {type(mask).__name__}"
        )
    raise TypeError(f"Unexpected model output type: {type(result).__name__}")


def remove_background(
    input_path: str,
    output_path: str,
    model_name: str = DEFAULT_MODEL,
    revision: str = DEFAULT_REVISION,
) -> int:
    """Remove background from images. Returns 0 on success, 1 on partial failure, -1 on no input."""
    input_dir = Path(input_path)
    output_dir = Path(output_path)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: input directory '{input_dir}' does not exist or is not a directory.")
        return -1

    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = _load_pipeline(model_name, revision)

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
            result = pipe(image)
            result_image = _extract_output_image(result)

            relative_path = image_file.relative_to(input_dir)
            output_file = output_dir / relative_path.with_name(
                f"{relative_path.stem}_no_bg.png"
            )
            output_file.parent.mkdir(parents=True, exist_ok=True)
            result_image.save(output_file)
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
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model name (default: briaai/RMBG-2.0)",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Git revision of the model to pin (default: main). Pin a commit SHA for reproducibility.",
    )
    args = parser.parse_args()
    sys.exit(remove_background(args.input, args.output, args.model, args.revision))
