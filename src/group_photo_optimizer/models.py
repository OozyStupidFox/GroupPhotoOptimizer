from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class ModelSpec:
    filename: str
    url: str
    min_bytes: int


MODEL_SPECS: Dict[str, ModelSpec] = {
    "yunet": ModelSpec(
        filename="face_detection_yunet_2023mar.onnx",
        url=(
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        min_bytes=100_000,
    ),
    "sface": ModelSpec(
        filename="face_recognition_sface_2021dec.onnx",
        url=(
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_recognition_sface/face_recognition_sface_2021dec.onnx"
        ),
        min_bytes=10_000_000,
    ),
    "face_landmarker": ModelSpec(
        filename="face_landmarker.task",
        url=(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task"
        ),
        min_bytes=3_000_000,
    ),
}


def download_models(models_dir: Path) -> Dict[str, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Path] = {}
    for name, spec in MODEL_SPECS.items():
        destination = models_dir / spec.filename
        if destination.exists() and destination.stat().st_size >= spec.min_bytes:
            result[name] = destination
            continue
        temporary = destination.with_suffix(destination.suffix + ".download")
        print("Downloading {} ...".format(spec.filename), flush=True)
        request = urllib.request.Request(spec.url, headers={"User-Agent": "group-photo-optimizer/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if temporary.stat().st_size < spec.min_bytes:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("Downloaded model is unexpectedly small: {}".format(spec.url))
        temporary.replace(destination)
        result[name] = destination
    return result


def resolve_models(models_dir: Path) -> Dict[str, Path]:
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        bundled_dir = Path(bundle_root) / "models"
        bundled = {
            name: bundled_dir / spec.filename for name, spec in MODEL_SPECS.items()
        }
        if all(
            path.exists() and path.stat().st_size >= MODEL_SPECS[name].min_bytes
            for name, path in bundled.items()
        ):
            return bundled
    return download_models(models_dir)


def model_fingerprint(paths: Dict[str, Path]) -> str:
    digest = hashlib.sha256()
    for name in sorted(paths):
        path = paths[name]
        digest.update(name.encode("ascii"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()[:16]
