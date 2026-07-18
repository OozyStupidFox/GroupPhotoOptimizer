from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .config import ManualConfig, ScoringConfig
from .types import FaceObservation, ImageAnalysis, PersonTrack


EXPRESSION_KEYS = (
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "jawOpen",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRollLower",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
)


def _blend(observation: FaceObservation, name: str) -> float:
    return float(observation.blendshapes.get(name, 0.0))


def _expression_vector(observation: FaceObservation) -> np.ndarray:
    return np.array([_blend(observation, key) for key in EXPRESSION_KEYS], dtype=np.float32)


def _base_metrics(observation: FaceObservation) -> None:
    blink_left = _blend(observation, "eyeBlinkLeft")
    blink_right = _blend(observation, "eyeBlinkRight")
    observation.eye_blink = max(blink_left, blink_right)
    observation.eye_asymmetry = abs(blink_left - blink_right)
    pairs = [
        (_blend(observation, "mouthPressLeft") + _blend(observation, "mouthPressRight")) / 2.0,
        (_blend(observation, "mouthFrownLeft") + _blend(observation, "mouthFrownRight")) / 2.0,
        _blend(observation, "mouthPucker"),
        _blend(observation, "mouthFunnel"),
        _blend(observation, "mouthRollLower"),
        _blend(observation, "mouthShrugLower"),
        0.72 * _blend(observation, "jawOpen"),
    ]
    observation.mouth_abnormal = max(pairs)


def score_tracks(tracks: Sequence[PersonTrack], config: ScoringConfig) -> None:
    for track in tracks:
        observations = list(track.observations.values())
        for observation in observations:
            _base_metrics(observation)
        valid = [item for item in observations if item.mesh is not None and item.blendshapes]
        if valid:
            vectors = np.stack([_expression_vector(item) for item in valid])
            median = np.median(vectors, axis=0)
            mad = np.median(np.abs(vectors - median), axis=0)
            # A floor keeps tiny numerical variations from becoming large z-scores.
            scale = np.maximum(1.4826 * mad, 0.075)
            for item, vector in zip(valid, vectors):
                item.expression_deviation = float(np.sqrt(np.mean(((vector - median) / scale) ** 2)))

        for observation in observations:
            has_analysis = observation.mesh is not None and bool(observation.blendshapes)
            observation.eyes_good = bool(
                has_analysis
                and observation.eye_blink < config.blink_threshold
                and observation.eye_asymmetry < config.eye_asymmetry_threshold
            )
            observation.mouth_good = bool(
                has_analysis
                and observation.mouth_abnormal < config.mouth_abnormal_threshold
                and observation.expression_deviation < config.expression_deviation_threshold
            )
            observation.expression_good = observation.eyes_good and observation.mouth_good
            observation.quality_good = bool(
                observation.detection_score >= config.min_detection_score
                and observation.sharpness >= config.min_sharpness
            )
            sharpness_score = float(np.clip(np.log1p(observation.sharpness) / np.log(301.0), 0.0, 1.0))
            observation.quality_score = float(
                0.35 * observation.detection_score
                + 0.25 * sharpness_score
                + 0.25 * (1.0 - observation.eye_blink)
                + 0.15 * (1.0 - observation.mouth_abnormal)
            )
            observation.overall_good = observation.expression_good and observation.quality_good


def image_score(image: ImageAnalysis) -> Tuple[int, int, int, float]:
    good = sum(item.overall_good for item in image.observations)
    open_eyes = sum(item.eyes_good for item in image.observations)
    analyzed = sum(item.mesh is not None for item in image.observations)
    quality = sum(item.quality_score for item in image.observations)
    return int(good), int(open_eyes), int(analyzed), float(quality)


def choose_base_image(
    images: Sequence[ImageAnalysis], manual: ManualConfig
) -> ImageAnalysis:
    if manual.base_image:
        for image in images:
            if image.path.name.casefold() == manual.base_image.casefold():
                return image
        raise ValueError("manual.base_image does not exist: {}".format(manual.base_image))
    return max(images, key=image_score)
