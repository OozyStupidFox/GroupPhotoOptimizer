from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class FaceObservation:
    image_index: int
    image_name: str
    bbox: np.ndarray
    landmarks5: np.ndarray
    detection_score: float
    embedding: np.ndarray
    sharpness: float
    mesh: Optional[np.ndarray] = None
    blendshapes: Dict[str, float] = field(default_factory=dict)
    canonical_center: Optional[np.ndarray] = None
    canonical_size: float = 0.0
    track_id: Optional[str] = None
    eye_blink: float = 1.0
    eye_asymmetry: float = 1.0
    mouth_abnormal: float = 1.0
    expression_deviation: float = 99.0
    eyes_good: bool = False
    mouth_good: bool = False
    expression_good: bool = False
    quality_good: bool = False
    overall_good: bool = False
    quality_score: float = 0.0

    @property
    def center(self) -> np.ndarray:
        x, y, w, h = self.bbox
        return np.array([x + w / 2.0, y + h / 2.0], dtype=np.float32)

    @property
    def size(self) -> float:
        return float((self.bbox[2] + self.bbox[3]) / 2.0)


@dataclass
class ImageAnalysis:
    index: int
    path: Path
    width: int
    height: int
    observations: List[FaceObservation] = field(default_factory=list)
    homography_to_canonical: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    alignment_inliers: int = 0
    alignment_ok: bool = True


@dataclass
class PersonTrack:
    track_id: str
    observations: Dict[int, FaceObservation] = field(default_factory=dict)
    canonical_center: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    canonical_size: float = 1.0
    embedding_template: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    def add(self, obs: FaceObservation) -> None:
        self.observations[obs.image_index] = obs
        obs.track_id = self.track_id


@dataclass
class ReplacementRecord:
    track_id: str
    region: str
    base_image: str
    donor_image: Optional[str]
    status: str
    identity_similarity: float = 0.0
    outside_similarity: float = 0.0
    reason: str = ""
    candidate_rejections: Dict[str, int] = field(default_factory=dict)
