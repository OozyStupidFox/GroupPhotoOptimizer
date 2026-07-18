from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DetectionConfig:
    tile_size: int = 1600
    tile_overlap: float = 0.30
    score_threshold: float = 0.72
    nms_threshold: float = 0.30
    min_face_size: int = 24
    crop_margin: float = 0.55


@dataclass
class AlignmentConfig:
    work_width: int = 1800
    min_matches: int = 35
    ransac_threshold: float = 3.0


@dataclass
class TrackingConfig:
    match_similarity: float = 0.42
    replacement_similarity: float = 0.52
    max_position_in_face_widths: float = 1.15
    min_observations_for_new_track: int = 3


@dataclass
class ScoringConfig:
    min_detection_score: float = 0.74
    min_sharpness: float = 55.0
    blink_threshold: float = 0.52
    eye_asymmetry_threshold: float = 0.42
    mouth_abnormal_threshold: float = 0.58
    expression_deviation_threshold: float = 3.5


@dataclass
class ReplacementConfig:
    min_outside_similarity: float = 0.64
    min_quality_ratio: float = 0.82
    min_expression_improvement: float = 0.12
    max_donor_blink: float = 0.48
    max_donor_mouth_abnormal: float = 0.50
    feather_ratio: float = 0.055
    max_replacements: int = 100
    replace_mouth: bool = True


@dataclass
class ManualConfig:
    base_image: Optional[str] = None
    exclude_tracks: List[str] = field(default_factory=list)
    force_donors: Dict[str, Dict[str, str]] = field(default_factory=dict)


@dataclass
class AppConfig:
    input_dir: str = "images"
    output_dir: str = "output"
    models_dir: str = "models"
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    replacement: ReplacementConfig = field(default_factory=ReplacementConfig)
    manual: ManualConfig = field(default_factory=ManualConfig)

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls(
            input_dir=raw.get("input_dir", "images"),
            output_dir=raw.get("output_dir", "output"),
            models_dir=raw.get("models_dir", "models"),
            detection=DetectionConfig(**raw.get("detection", {})),
            alignment=AlignmentConfig(**raw.get("alignment", {})),
            tracking=TrackingConfig(**raw.get("tracking", {})),
            scoring=ScoringConfig(**raw.get("scoring", {})),
            replacement=ReplacementConfig(**raw.get("replacement", {})),
            manual=ManualConfig(**raw.get("manual", {})),
        )

    def resolve_paths(self, config_path: Path) -> "ResolvedPaths":
        root = config_path.resolve().parent
        return ResolvedPaths(
            root=root,
            input_dir=(root / self.input_dir).resolve(),
            output_dir=(root / self.output_dir).resolve(),
            models_dir=(root / self.models_dir).resolve(),
        )


@dataclass(frozen=True)
class ResolvedPaths:
    root: Path
    input_dir: Path
    output_dir: Path
    models_dir: Path


def to_plain_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_plain_dict(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {key: to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_dict(item) for item in value]
    return value
