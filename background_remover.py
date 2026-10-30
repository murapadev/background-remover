#!/usr/bin/env python3
"""Remove the background from every image in a folder with a Hugging Face model.

Author: Pablo R (murapadev)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

from PIL import Image, ImageChops, ImageOps

# Native transformers architecture (no remote code), MIT licensed.
DEFAULT_MODEL = "tue-mps/coco_instance_eomt_large_640"
DEFAULT_REVISION = "a43666870d47a7010bfea2c4554f7ca829714558"
DEFAULT_FORMATS = ("jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp")
OUTPUT_SUFFIX = "_no_bg"

Pipeline = Callable[[Image.Image], object]


class NoForegroundError(ValueError):
    pass


def select_device(requested: str = "auto") -> str:
    """Resolve "auto" to cuda, mps or cpu depending on what torch can actually use."""
    if requested != "auto":
        return requested

    import torch

    if torch.cuda.is_available():
        # Recent CUDA wheels drop old GPUs (e.g. Pascal): is_available() is True
        # but every kernel launch fails, so check the build supports this card.
        major, minor = torch.cuda.get_device_capability(0)
        arches = torch.cuda.get_arch_list()
        # PTX (compute_XY) can be JIT-compiled for any GPU at least that new.
        ptx = [int(a[8:]) for a in arches if a.startswith("compute_") and a[8:].isdigit()]
        if f"sm_{major}{minor}" in arches or any(v <= major * 10 + minor for v in ptx):
            return "cuda"
        print(
            f"Warning: this torch build does not support the GPU "
            f"(sm_{major}{minor}); falling back to CPU.",
            file=sys.stderr,
        )
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def load_pipeline(
    model: str,
    revision: str | None,
    device: str,
    trust_remote_code: bool = False,
) -> Pipeline:
    from transformers import pipeline

    return pipeline(
        "image-segmentation",
        model=model,
        revision=revision,
        device=device,
        trust_remote_code=trust_remote_code,
    )


def _union_masks(masks: Iterable[Image.Image], size: tuple[int, int]) -> Image.Image:
    alpha = Image.new("L", size, 0)
    for mask in masks:
        mask = mask.convert("L")
        if mask.size != size:
            mask = mask.resize(size, Image.Resampling.BILINEAR)
        alpha = ImageChops.lighter(alpha, mask)
    return alpha


def to_rgba(
    image: Image.Image,
    result: object,
    labels: Iterable[str] | None = None,
    ignore_labels: Iterable[str] | None = None,
) -> Image.Image:
    """Build an RGBA cutout from whatever the segmentation pipeline returned.

    - PIL RGBA/LA image: custom pipelines (e.g. RMBG-1.4) that already return the cutout.
    - PIL L/1 image: a foreground mask.
    - list of {"label", "score", "mask"} dicts: the stock transformers pipeline;
      the foreground is the union of the masks kept by ``labels``/``ignore_labels``.
    """
    rgb = image.convert("RGB")

    if isinstance(result, dict):
        result = [result]

    if isinstance(result, Image.Image):
        if result.mode in ("RGBA", "LA"):
            masks = [result.getchannel("A")]
        elif result.mode in ("L", "1"):
            masks = [result]
        else:
            raise TypeError(f"Cannot use a {result.mode} image as a mask")
    elif isinstance(result, list):
        keep = {label.lower() for label in labels} if labels else None
        drop = {label.lower() for label in ignore_labels or ()}
        masks = []
        for segment in result:
            if not isinstance(segment, dict) or "mask" not in segment:
                raise TypeError(f"Unexpected segment in pipeline output: {segment!r}")
            label = str(segment.get("label") or "").lower()
            if (keep is not None and label not in keep) or label in drop:
                continue
            if not isinstance(segment["mask"], Image.Image):
                raise TypeError(f"Expected a PIL mask, got {type(segment['mask']).__name__}")
            masks.append(segment["mask"])
        if not masks:
            found = ", ".join(sorted({str(s.get("label")) for s in result})) or "nothing"
            raise NoForegroundError(f"no foreground detected (found: {found})")
    else:
        raise TypeError(f"Unexpected pipeline output type: {type(result).__name__}")

    rgba = rgb.copy()
    rgba.putalpha(_union_masks(masks, rgb.size))
    return rgba


def find_images(
    input_dir: Path,
    formats: Iterable[str],
    recursive: bool,
    exclude: Path | None = None,
) -> list[Path]:
    extensions = {f".{fmt.lower().lstrip('.')}" for fmt in formats}
    candidates = input_dir.rglob("*") if recursive else input_dir.iterdir()
    exclude = exclude.resolve() if exclude else None
    images = []
    for path in candidates:
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if exclude and exclude in path.resolve().parents:
            continue
        images.append(path)
    return sorted(images)


def output_path_for(image_file: Path, input_dir: Path, output_dir: Path) -> Path:
    relative = image_file.relative_to(input_dir)
    return output_dir / relative.with_name(f"{relative.stem}{OUTPUT_SUFFIX}.png")


def process_image(
    pipe: Pipeline,
    image_file: Path,
    output_file: Path,
    labels: Iterable[str] | None = None,
    ignore_labels: Iterable[str] | None = None,
) -> None:
    with Image.open(image_file) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    cutout = to_rgba(image, pipe(image), labels, ignore_labels)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cutout.save(output_file, format="PNG")


def process_batch(
    pipe: Pipeline,
    images: list[Path],
    input_dir: Path,
    output_dir: Path,
    labels: Iterable[str] | None = None,
    ignore_labels: Iterable[str] | None = None,
) -> list[tuple[Path, str]]:
    """Process every image; return (path, error) for the ones that failed."""
    failures = []
    for index, image_file in enumerate(images, start=1):
        output_file = output_path_for(image_file, input_dir, output_dir)
        print(f"[{index}/{len(images)}] {image_file.relative_to(input_dir)}")
        try:
            process_image(pipe, image_file, output_file, labels, ignore_labels)
        except Exception as exc:  # noqa: BLE001 - one bad image must not abort the batch
            message = f"{type(exc).__name__}: {exc}"
            print(f"  error: {message}", file=sys.stderr)
            failures.append((image_file, message))
        else:
            print(f"  -> {output_file}")
    return failures


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove the background from every image in a folder and save transparent PNGs.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("input"),
        help="folder with the source images (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output"),
        help="folder for the PNG cutouts (default: %(default)s)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model id (default: %(default)s)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="model branch, tag or commit; the default model is pinned to a "
        "verified commit, other models default to their main branch",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="also process images in subfolders, mirroring the tree in the output",
    )
    parser.add_argument(
        "-f",
        "--formats",
        type=_csv,
        default=",".join(DEFAULT_FORMATS),
        help="comma-separated extensions to process (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="where to run the model (default: auto = GPU if usable, else CPU)",
    )
    parser.add_argument(
        "--labels",
        type=_csv,
        default=None,
        help="only keep these segment labels, e.g. 'person,cat' (stock pipeline models only)",
    )
    parser.add_argument(
        "--ignore-labels",
        type=_csv,
        default=None,
        help="drop these segment labels, e.g. 'dining table' (stock pipeline models only)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="allow models that ship their own Python code (e.g. briaai/RMBG-1.4); "
        "that code runs on your machine, only enable it for repos you trust",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Exit codes: 0 all images processed, 1 some failed, 2 bad input."""
    args = build_parser().parse_args(argv)
    input_dir: Path = args.input
    output_dir: Path = args.output

    if not input_dir.is_dir():
        print(f"error: input folder '{input_dir}' does not exist", file=sys.stderr)
        return 2

    images = find_images(input_dir, args.formats, args.recursive, exclude=output_dir)
    if not images:
        print(
            f"error: no images with extensions {', '.join(args.formats)} in '{input_dir}'",
            file=sys.stderr,
        )
        return 2

    revision = args.revision
    if revision is None and args.model == DEFAULT_MODEL:
        revision = DEFAULT_REVISION

    device = select_device(args.device)
    print(f"Loading {args.model} on {device}...")
    try:
        pipe = load_pipeline(args.model, revision, device, args.trust_remote_code)
    except Exception as exc:  # noqa: BLE001 - report any load failure without a traceback
        print(f"error: could not load model '{args.model}': {exc}", file=sys.stderr)
        if "trust_remote_code" in str(exc):
            print("hint: this model needs --trust-remote-code", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    failures = process_batch(pipe, images, input_dir, output_dir, args.labels, args.ignore_labels)

    print(f"\nDone: {len(images) - len(failures)}/{len(images)} images processed.")
    for path, message in failures:
        print(f"  failed: {path} ({message})", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
