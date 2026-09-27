import argparse
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from zipfile import ZipFile

import numpy as np
import SimpleITK as sitk
import yaml


def read_landmarks(filepath: Path, sep: str | None = None) -> np.ndarray:
    possible_seps = (" ", "\t", ",")
    lines = filepath.read_text().splitlines()
    if not lines:
        return np.empty((0, 3), dtype=np.float32)

    if sep is None:
        for candidate in possible_seps:
            if candidate in lines[0]:
                sep = candidate
                break
        else:
            raise RuntimeError(f"Could not infer landmark separator for {filepath}")

    points = [
        tuple(map(float, line.strip().split(sep))) for line in lines if line.strip()
    ]
    return np.asarray(points, dtype=np.float32)


def write_landmarks(landmarks: np.ndarray, filepath: Path, sep: str = ",") -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(filepath, landmarks, delimiter=sep, fmt="%.3f")


def parse_meta(raw_meta: str) -> dict[str, Any]:
    decimal_regex = r"([0-9]*[.])?[0-9]+"
    lines = raw_meta.splitlines()

    dims_match = re.search(r"(\d+)\s*x\s*(\d+)\s*x\s*(\d+)", lines[0])
    spacing_match = re.search(
        rf"(?P<x>{decimal_regex})\s*x\s*"
        rf"(?P<y>{decimal_regex})\s*x\s*"
        rf"(?P<z>{decimal_regex})",
        lines[1],
    )

    if dims_match is None or spacing_match is None:
        raise ValueError(f"Could not parse metadata block:\n{raw_meta}")

    meta = {
        "image_shape": [int(val) for val in dims_match.groups()],
        "image_spacing": [
            float(spacing_match.groupdict()[axis]) for axis in ("x", "y", "z")
        ],
    }

    if len(lines) > 5:
        observer_match = re.search(
            rf"(?P<mean>{decimal_regex})\s\((?P<std>{decimal_regex})\)",
            lines[5],
        )
        if observer_match is not None:
            meta["observer_tre_mean"] = float(observer_match.groupdict()["mean"])
            meta["observer_tre_std"] = float(observer_match.groupdict()["std"])

    return meta


def read_raw_image(filepath: Path, meta: dict[str, Any]) -> sitk.Image:
    image = np.fromfile(filepath, dtype=np.int16)
    image = image - 1024
    image = np.clip(image, -1024, 3071)
    image = image.reshape(tuple(meta["image_shape"][::-1]))
    image = np.flip(image, axis=0)

    image_sitk = sitk.GetImageFromArray(image)
    image_sitk.SetSpacing(tuple(meta["image_spacing"]))
    return image_sitk


def flip_landmarks_z(landmarks: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    flipped = landmarks.copy()
    flipped[:, 2] = meta["image_shape"][2] - flipped[:, 2]
    return flipped


def save_metadata(output_folder: Path, meta: dict[str, Any]) -> None:
    with (output_folder / "metadata.yaml").open("w") as handle:
        yaml.safe_dump(meta, handle, sort_keys=False)


def ensure_case_folders(case_folder: Path) -> dict[str, Path]:
    paths = {
        "case": case_folder,
        "images": case_folder / "images",
        "landmarks_raw": case_folder / "landmarks_raw",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def segment_with_totalsegmentator(
    image_path: Path,
    segmentation_path: Path,
    *,
    device: str,
    task: str = "total",
    multilabel: bool = True,
    probabilities_path: Path | None = None,
) -> None:
    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError as exc:
        raise ImportError(
            "TotalSegmentator is not installed. Install the optional dependency with "
            "`uv sync --extra segmentation` or `pip install -e .[segmentation]`."
        ) from exc

    segmentation_path.parent.mkdir(parents=True, exist_ok=True)
    if probabilities_path is not None:
        probabilities_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Segmenting {image_path.name} with TotalSegmentator task {task!r}")
    segmentation_nifti = totalsegmentator(
        input=image_path,
        output=segmentation_path,
        ml=multilabel,
        task=task,
        quiet=True,
        device=device,
        save_probabilities=probabilities_path,
    )

    if multilabel and not segmentation_path.exists():
        try:
            import nibabel as nib
        except ImportError as exc:
            raise RuntimeError(
                "TotalSegmentator returned a NIfTI image but did not save it, and "
                "nibabel is unavailable to persist the segmentation."
            ) from exc
        nib.save(segmentation_nifti, str(segmentation_path))


def extract_archive(archive_path: Path, tmp_dir: Path) -> None:
    with ZipFile(archive_path, "r") as archive:
        archive.extractall(tmp_dir)


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("input_folder", type=Path)
    parser.add_argument("--output-folder", type=Path, default=None)
    parser.add_argument(
        "--use-totalsegmentator",
        action="store_true",
        help=(
            "Generate lung-lobe, body, and lung-vessel segmentations required by "
            "the training and evaluation pipeline."
        ),
    )
    parser.add_argument("--totalsegmentator-device", type=str, default="gpu")
    return parser


def resolve_output_folder(input_folder: Path, output_folder: Path | None) -> Path:
    return (
        output_folder
        if output_folder is not None
        else input_folder.parent / "converted"
    )


def temporary_directory() -> TemporaryDirectory[str]:
    return TemporaryDirectory()
