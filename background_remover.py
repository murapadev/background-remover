#!/usr/bin/env python3
"""
Background Remover Script
Author: Pablo R (murapadev)
Description: Removes backgrounds from images in the input folder using a Hugging Face model.
"""

import os
import argparse
from pathlib import Path
from transformers import pipeline
from PIL import Image

def remove_background(input_path, output_path, model_name="briaai/RMBG-1.4"):
    """
    Remove background from images in input_path and save to output_path.
    """
    # Load the model
    pipe = pipeline("image-segmentation", model=model_name, trust_remote_code=True)

    input_dir = Path(input_path)
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True)

    # Supported image formats
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    for image_file in input_dir.iterdir():
        if image_file.suffix.lower() in image_extensions:
            print(f"Processing {image_file.name}...")
            try:
                # Load image
                image = Image.open(image_file)

                # Remove background
                result = pipe(image)

                # Save the result (assuming the pipeline returns a PIL Image)
                output_file = output_dir / f"{image_file.stem}_no_bg.png"
                result.save(output_file)
                print(f"Saved {output_file}")

            except Exception as e:
                print(f"Error processing {image_file.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove backgrounds from images.")
    parser.add_argument("--input", default="./input", help="Input folder with images")
    parser.add_argument("--output", default="./output", help="Output folder for processed images")
    parser.add_argument("--model", default="briaai/RMBG-1.4", help="Hugging Face model name")

    args = parser.parse_args()
    remove_background(args.input, args.output, args.model)