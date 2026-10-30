"""Tests run without torch/transformers: the pipeline is always mocked."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from PIL import Image

import background_remover as br


def _mask(size: tuple[int, int], box: tuple[int, int, int, int]) -> Image.Image:
    mask = Image.new("L", size, 0)
    mask.paste(255, box)
    return mask


def _segments(size: tuple[int, int]) -> list[dict]:
    """Shape of the stock transformers image-segmentation pipeline output."""
    return [
        {"label": "person", "score": 0.99, "mask": _mask(size, (0, 0, 4, 4))},
        {"label": "dining table", "score": 0.9, "mask": _mask(size, (4, 4, 8, 8))},
    ]


@pytest.fixture
def image() -> Image.Image:
    return Image.new("RGB", (8, 8), (200, 10, 10))


def alpha_at(img: Image.Image, xy: tuple[int, int]) -> int:
    return img.getchannel("A").getpixel(xy)


class TestToRgba:
    def test_list_output_becomes_transparent_cutout(self, image):
        # Regression: the list returned by the pipeline used to be .save()d directly.
        out = br.to_rgba(image, _segments(image.size))
        assert out.mode == "RGBA"
        assert out.size == image.size
        assert alpha_at(out, (1, 1)) == 255
        assert alpha_at(out, (6, 6)) == 255
        assert alpha_at(out, (6, 1)) == 0
        assert out.getpixel((1, 1))[:3] == (200, 10, 10)

    def test_labels_keep_only_selected_segments(self, image):
        out = br.to_rgba(image, _segments(image.size), labels=["Person"])
        assert alpha_at(out, (1, 1)) == 255
        assert alpha_at(out, (6, 6)) == 0

    def test_ignore_labels_drops_segments(self, image):
        out = br.to_rgba(image, _segments(image.size), ignore_labels=["dining table"])
        assert alpha_at(out, (1, 1)) == 255
        assert alpha_at(out, (6, 6)) == 0

    def test_single_dict(self, image):
        out = br.to_rgba(image, _segments(image.size)[0])
        assert alpha_at(out, (1, 1)) == 255

    def test_mask_image(self, image):
        out = br.to_rgba(image, _mask(image.size, (0, 0, 2, 2)))
        assert alpha_at(out, (0, 0)) == 255
        assert alpha_at(out, (5, 5)) == 0

    def test_rgba_cutout_from_custom_pipeline(self, image):
        cutout = image.copy()
        cutout.putalpha(_mask(image.size, (2, 2, 6, 6)))
        out = br.to_rgba(image, cutout)
        assert alpha_at(out, (3, 3)) == 255
        assert alpha_at(out, (0, 0)) == 0

    def test_mask_with_different_size_is_resized(self, image):
        out = br.to_rgba(image, _mask((4, 4), (0, 0, 2, 2)))
        assert out.size == image.size
        assert alpha_at(out, (1, 1)) == 255
        assert alpha_at(out, (7, 7)) == 0

    def test_nothing_detected(self, image):
        with pytest.raises(br.NoForegroundError):
            br.to_rgba(image, [])

    def test_everything_filtered_out(self, image):
        with pytest.raises(br.NoForegroundError, match="person"):
            br.to_rgba(image, _segments(image.size), labels=["cat"])

    @pytest.mark.parametrize(
        "result",
        [None, "mask", Image.new("RGB", (8, 8)), [{"label": "x", "mask": "nope"}], [42]],
    )
    def test_unexpected_output(self, image, result):
        with pytest.raises(TypeError):
            br.to_rgba(image, result)


def _write(path: Path, data: bytes | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if data is None:
        Image.new("RGB", (8, 8), (0, 0, 255)).save(path)
    else:
        path.write_bytes(data)
    return path


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "in"
    _write(root / "a.jpg")
    _write(root / "b.PNG")
    _write(root / "notes.txt", b"hello")
    _write(root / "sub" / "c.webp")
    return root


class TestFindImages:
    def test_flat(self, tree):
        found = br.find_images(tree, br.DEFAULT_FORMATS, recursive=False)
        assert [p.name for p in found] == ["a.jpg", "b.PNG"]

    def test_recursive(self, tree):
        found = br.find_images(tree, br.DEFAULT_FORMATS, recursive=True)
        assert [p.relative_to(tree).as_posix() for p in found] == [
            "a.jpg",
            "b.PNG",
            "sub/c.webp",
        ]

    def test_formats(self, tree):
        found = br.find_images(tree, [".webp"], recursive=True)
        assert [p.name for p in found] == ["c.webp"]

    def test_output_inside_input_is_skipped(self, tree):
        found = br.find_images(tree, br.DEFAULT_FORMATS, recursive=True, exclude=tree / "sub")
        assert "c.webp" not in {p.name for p in found}


@pytest.fixture
def fake_pipeline(monkeypatch):
    calls: dict = {}

    def pipe(img: Image.Image):
        return _segments(img.size)[:1]

    def fake_load(model, revision, device, trust_remote_code=False):
        calls.update(model=model, revision=revision, device=device, trust=trust_remote_code)
        return pipe

    monkeypatch.setattr(br, "load_pipeline", fake_load)
    monkeypatch.setattr(br, "select_device", lambda requested="auto": "cpu")
    return calls


class TestMain:
    def test_processes_batch_and_writes_rgba_png(self, tree, tmp_path, fake_pipeline):
        out = tmp_path / "out"
        assert br.main(["-i", str(tree), "-o", str(out), "-r"]) == 0
        results = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.png"))
        assert results == ["a_no_bg.png", "b_no_bg.png", "sub/c_no_bg.png"]
        with Image.open(out / "a_no_bg.png") as img:
            assert img.format == "PNG"
            assert img.mode == "RGBA"
            assert alpha_at(img, (1, 1)) == 255
            assert alpha_at(img, (6, 6)) == 0

    def test_default_model_is_pinned_and_safe(self, tree, tmp_path, fake_pipeline):
        br.main(["-i", str(tree), "-o", str(tmp_path / "out")])
        assert fake_pipeline == {
            "model": br.DEFAULT_MODEL,
            "revision": br.DEFAULT_REVISION,
            "device": "cpu",
            "trust": False,
        }

    def test_custom_model_options(self, tree, tmp_path, fake_pipeline):
        br.main(
            [
                "-i",
                str(tree),
                "-o",
                str(tmp_path / "out"),
                "-m",
                "briaai/RMBG-1.4",
                "--revision",
                "abc123",
                "--trust-remote-code",
            ]
        )
        assert fake_pipeline["model"] == "briaai/RMBG-1.4"
        assert fake_pipeline["revision"] == "abc123"
        assert fake_pipeline["trust"] is True

    def test_custom_model_without_revision_uses_main(self, tree, tmp_path, fake_pipeline):
        br.main(["-i", str(tree), "-o", str(tmp_path / "out"), "-m", "some/model"])
        assert fake_pipeline["revision"] is None

    def test_bad_image_does_not_abort_batch(self, tree, tmp_path, fake_pipeline, capsys):
        _write(tree / "broken.jpg", b"not an image")
        out = tmp_path / "out"
        assert br.main(["-i", str(tree), "-o", str(out)]) == 1
        assert {p.name for p in out.glob("*.png")} == {"a_no_bg.png", "b_no_bg.png"}
        assert "broken.jpg" in capsys.readouterr().err

    def test_pipeline_error_does_not_abort_batch(self, tree, tmp_path, monkeypatch):
        calls = []

        def flaky(img):
            calls.append(img)
            if len(calls) == 1:
                raise RuntimeError("CUDA out of memory")
            return _mask(img.size, (0, 0, 2, 2))

        monkeypatch.setattr(br, "load_pipeline", lambda *a, **k: flaky)
        monkeypatch.setattr(br, "select_device", lambda requested="auto": "cpu")
        out = tmp_path / "out"
        assert br.main(["-i", str(tree), "-o", str(out)]) == 1
        assert len(list(out.glob("*.png"))) == 1

    def test_missing_input(self, tmp_path, fake_pipeline):
        assert br.main(["-i", str(tmp_path / "nope")]) == 2
        assert fake_pipeline == {}

    def test_no_images(self, tmp_path, fake_pipeline):
        (tmp_path / "empty").mkdir()
        assert br.main(["-i", str(tmp_path / "empty")]) == 2

    def test_model_load_failure(self, tree, tmp_path, monkeypatch, capsys):
        def boom(*args, **kwargs):
            raise ValueError("please pass trust_remote_code=True")

        monkeypatch.setattr(br, "load_pipeline", boom)
        monkeypatch.setattr(br, "select_device", lambda requested="auto": "cpu")
        assert br.main(["-i", str(tree), "-o", str(tmp_path / "out")]) == 2
        assert "--trust-remote-code" in capsys.readouterr().err


def _fake_torch(cuda=False, capability=(8, 9), arches=("sm_89",), mps=False):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda,
        get_device_capability=lambda index=0: capability,
        get_arch_list=lambda: list(arches),
    )
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    return torch


class TestSelectDevice:
    def test_explicit_device_is_respected(self):
        assert br.select_device("cpu") == "cpu"

    @pytest.mark.parametrize(
        ("torch", "expected"),
        [
            (_fake_torch(cuda=True), "cuda"),
            (_fake_torch(cuda=True, capability=(6, 1), arches=("sm_75", "sm_89")), "cpu"),
            (_fake_torch(cuda=True, capability=(12, 0), arches=("sm_90", "compute_90")), "cuda"),
            (_fake_torch(mps=True), "mps"),
            (_fake_torch(), "cpu"),
        ],
    )
    def test_auto(self, monkeypatch, torch, expected):
        monkeypatch.setitem(sys.modules, "torch", torch)
        assert br.select_device("auto") == expected
