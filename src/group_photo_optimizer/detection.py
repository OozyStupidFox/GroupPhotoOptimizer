from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .config import DetectionConfig
from .types import FaceObservation


def _tile_starts(length: int, tile_size: int, overlap: float) -> List[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, int(tile_size * (1.0 - overlap)))
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _iou(one: np.ndarray, many: np.ndarray) -> np.ndarray:
    x1 = np.maximum(one[0], many[:, 0])
    y1 = np.maximum(one[1], many[:, 1])
    x2 = np.minimum(one[0] + one[2], many[:, 0] + many[:, 2])
    y2 = np.minimum(one[1] + one[3], many[:, 1] + many[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    union = one[2] * one[3] + many[:, 2] * many[:, 3] - intersection
    return intersection / np.maximum(union, 1e-6)


def _global_nms(faces: List[np.ndarray], threshold: float) -> List[np.ndarray]:
    if not faces:
        return []
    ordered = sorted(faces, key=lambda row: float(row[-1]), reverse=True)
    kept: List[np.ndarray] = []
    while ordered:
        best = ordered.pop(0)
        kept.append(best)
        if not ordered:
            break
        remaining = np.stack(ordered)
        keep_mask = _iou(best[:4], remaining[:, :4]) <= threshold
        ordered = [row for row, keep in zip(ordered, keep_mask) if keep]
    return kept


def _expanded_crop(
    image: np.ndarray, bbox: np.ndarray, margin: float
) -> Tuple[np.ndarray, Tuple[int, int]]:
    x, y, w, h = bbox.astype(float)
    side = max(w, h) * (1.0 + 2.0 * margin)
    cx, cy = x + w / 2.0, y + h / 2.0
    x0 = max(0, int(round(cx - side / 2.0)))
    y0 = max(0, int(round(cy - side / 2.0)))
    x1 = min(image.shape[1], int(round(cx + side / 2.0)))
    y1 = min(image.shape[0], int(round(cy + side / 2.0)))
    return image[y0:y1, x0:x1], (x0, y0)


def _sharpness(image: np.ndarray, bbox: np.ndarray) -> float:
    x, y, w, h = bbox.astype(int)
    x, y = max(x, 0), max(y, 0)
    crop = image[y : y + max(h, 1), x : x + max(w, 1)]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class FaceMeshAnalyzer(AbstractContextManager):
    def __init__(self, model_path: Path):
        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.25,
            min_face_presence_confidence=0.25,
            min_tracking_confidence=0.25,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        self._landmarker.close()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def analyze(
        self, image: np.ndarray, bbox: np.ndarray, margin: float
    ) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
        crop, (x0, y0) = _expanded_crop(image, bbox, margin)
        if crop.size == 0:
            return None, {}
        # Upscaling is important because rear-row faces can be only 30-40 pixels wide.
        scale = max(1.0, 320.0 / max(crop.shape[:2]))
        if scale > 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.face_landmarks:
            return None, {}
        points = np.array(
            [[landmark.x * crop.shape[1] / scale + x0, landmark.y * crop.shape[0] / scale + y0]
             for landmark in result.face_landmarks[0]],
            dtype=np.float32,
        )
        blendshapes: Dict[str, float] = {}
        if result.face_blendshapes:
            blendshapes = {
                category.category_name: float(category.score)
                for category in result.face_blendshapes[0]
            }
        return points, blendshapes


class FaceDetector:
    def __init__(
        self,
        yunet_path: Path,
        sface_path: Path,
        landmarker_path: Path,
        config: DetectionConfig,
    ):
        self.config = config
        self.detector = cv2.FaceDetectorYN.create(
            str(yunet_path),
            "",
            (config.tile_size, config.tile_size),
            config.score_threshold,
            config.nms_threshold,
            5000,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(sface_path), "")
        self.mesh_analyzer = FaceMeshAnalyzer(landmarker_path)

    def close(self) -> None:
        self.mesh_analyzer.close()

    def _detect_rows(self, image: np.ndarray) -> List[np.ndarray]:
        height, width = image.shape[:2]
        tile_size = self.config.tile_size
        rows: List[np.ndarray] = []
        for y0 in _tile_starts(height, tile_size, self.config.tile_overlap):
            for x0 in _tile_starts(width, tile_size, self.config.tile_overlap):
                tile = image[y0 : min(y0 + tile_size, height), x0 : min(x0 + tile_size, width)]
                self.detector.setInputSize((tile.shape[1], tile.shape[0]))
                _, detected = self.detector.detect(tile)
                if detected is None:
                    continue
                for row in detected:
                    row = row.astype(np.float32).copy()
                    row[[0, 4, 6, 8, 10, 12]] += x0
                    row[[1, 5, 7, 9, 11, 13]] += y0
                    if min(row[2], row[3]) >= self.config.min_face_size:
                        rows.append(row)
        return _global_nms(rows, self.config.nms_threshold)

    def analyze(self, image: np.ndarray, image_index: int, image_name: str) -> List[FaceObservation]:
        observations: List[FaceObservation] = []
        for row in self._detect_rows(image):
            try:
                aligned = self.recognizer.alignCrop(image, row)
                embedding = self.recognizer.feature(aligned).reshape(-1).astype(np.float32)
                embedding /= max(float(np.linalg.norm(embedding)), 1e-8)
            except cv2.error:
                continue
            bbox = row[:4].copy()
            mesh, blendshapes = self.mesh_analyzer.analyze(image, bbox, self.config.crop_margin)
            observations.append(
                FaceObservation(
                    image_index=image_index,
                    image_name=image_name,
                    bbox=bbox,
                    landmarks5=row[4:14].reshape(5, 2).copy(),
                    detection_score=float(row[-1]),
                    embedding=embedding,
                    sharpness=_sharpness(image, bbox),
                    mesh=mesh,
                    blendshapes=blendshapes,
                )
            )
        return observations


def cosine_similarity(one: np.ndarray, two: np.ndarray) -> float:
    if one.size == 0 or two.size == 0:
        return -1.0
    return float(np.dot(one, two) / max(np.linalg.norm(one) * np.linalg.norm(two), 1e-8))
