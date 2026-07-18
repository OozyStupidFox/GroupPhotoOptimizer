from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import ManualConfig, ReplacementConfig, TrackingConfig
from .detection import cosine_similarity
from .types import FaceObservation, ImageAnalysis, PersonTrack, ReplacementRecord


LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
ALIGN_FOR_EYES = [1, 4, 5, 6, 10, 61, 152, 168, 234, 291, 454]
ALIGN_FOR_MOUTH = [1, 4, 6, 10, 33, 133, 152, 168, 234, 263, 291, 362, 454]


def _region_mask(
    shape: Tuple[int, int], mesh: np.ndarray, region: str, face_size: float
) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    groups = [LEFT_EYE, RIGHT_EYE] if region == "eyes" else [MOUTH]
    dilation_ratio = 0.095 if region == "eyes" else 0.085
    for indices in groups:
        points = np.round(mesh[indices]).astype(np.int32)
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255, lineType=cv2.LINE_AA)
    radius = max(2, int(round(face_size * dilation_ratio)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask, kernel)


def _face_context_mask(shape: Tuple[int, int], observation: FaceObservation) -> np.ndarray:
    height, width = shape
    x, y, w, h = observation.bbox
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (int(round(x + w / 2.0)), int(round(y + h / 2.0)))
    axes = (max(1, int(round(w * 0.46))), max(1, int(round(h * 0.56))))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1, cv2.LINE_AA)
    return mask


def _roi_for_face(observation: FaceObservation, image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    height, width = image_shape
    x, y, w, h = observation.bbox
    margin = 0.48 * max(w, h)
    return (
        max(0, int(np.floor(x - margin))),
        max(0, int(np.floor(y - margin))),
        min(width, int(np.ceil(x + w + margin))),
        min(height, int(np.ceil(y + h + margin))),
    )


def _masked_ssim(one: np.ndarray, two: np.ndarray, mask: np.ndarray) -> float:
    one_gray = cv2.cvtColor(one, cv2.COLOR_BGR2GRAY).astype(np.float32)
    two_gray = cv2.cvtColor(two, cv2.COLOR_BGR2GRAY).astype(np.float32)
    selected = mask > 127
    if int(selected.sum()) < 30:
        return 0.0
    one_values, two_values = one_gray[selected], two_gray[selected]
    one_gray = (one_gray - float(one_values.mean())) / max(float(one_values.std()), 8.0)
    two_gray = (two_gray - float(two_values.mean())) / max(float(two_values.std()), 8.0)
    mu_one = cv2.GaussianBlur(one_gray, (0, 0), 1.2)
    mu_two = cv2.GaussianBlur(two_gray, (0, 0), 1.2)
    variance_one = cv2.GaussianBlur(one_gray * one_gray, (0, 0), 1.2) - mu_one * mu_one
    variance_two = cv2.GaussianBlur(two_gray * two_gray, (0, 0), 1.2) - mu_two * mu_two
    covariance = cv2.GaussianBlur(one_gray * two_gray, (0, 0), 1.2) - mu_one * mu_two
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_one * mu_two + c1) * (2 * covariance + c2)) / (
        (mu_one * mu_one + mu_two * mu_two + c1) * (variance_one + variance_two + c2) + 1e-8
    )
    return float(np.clip(np.mean(ssim[selected]), -1.0, 1.0))


def _color_match(donor: np.ndarray, target: np.ndarray, context_mask: np.ndarray) -> np.ndarray:
    selected = context_mask > 127
    if int(selected.sum()) < 30:
        return donor
    donor_lab = cv2.cvtColor(donor, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    for channel in range(3):
        donor_values = donor_lab[:, :, channel][selected]
        target_values = target_lab[:, :, channel][selected]
        scale = np.clip(
            float(target_values.std()) / max(float(donor_values.std()), 2.0), 0.82, 1.22
        )
        donor_lab[:, :, channel] = (
            donor_lab[:, :, channel] - float(donor_values.mean())
        ) * scale + float(target_values.mean())
    return cv2.cvtColor(np.clip(donor_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _candidate_score(observation: FaceObservation, region: str) -> float:
    expression = 1.0 - (observation.eye_blink if region == "eyes" else observation.mouth_abnormal)
    return 0.62 * expression + 0.38 * observation.quality_score


def _candidate_observations(
    track: PersonTrack,
    base: FaceObservation,
    region: str,
    tracking_config: TrackingConfig,
    replacement_config: ReplacementConfig,
    forced_name: Optional[str],
) -> Tuple[List[Tuple[FaceObservation, float]], Dict[str, int]]:
    result: List[Tuple[FaceObservation, float]] = []
    rejections: Counter = Counter()
    for candidate in track.observations.values():
        if candidate.image_index == base.image_index:
            continue
        if candidate.mesh is None:
            rejections["landmarks_unavailable"] += 1
            continue
        region_good = candidate.eyes_good if region == "eyes" else candidate.mouth_good
        if not region_good:
            rejections["expression_not_good"] += 1
            continue
        if not candidate.quality_good:
            rejections["quality_not_good"] += 1
            continue
        donor_metric = candidate.eye_blink if region == "eyes" else candidate.mouth_abnormal
        base_metric = base.eye_blink if region == "eyes" else base.mouth_abnormal
        maximum_donor_metric = (
            replacement_config.max_donor_blink
            if region == "eyes"
            else replacement_config.max_donor_mouth_abnormal
        )
        if donor_metric > maximum_donor_metric:
            rejections["donor_metric_above_limit"] += 1
            continue
        if base_metric - donor_metric < replacement_config.min_expression_improvement:
            rejections["improvement_too_small"] += 1
            continue
        if forced_name and candidate.image_name.casefold() != forced_name.casefold():
            rejections["not_forced_donor"] += 1
            continue
        base_similarity = cosine_similarity(base.embedding, candidate.embedding)
        template_similarity = cosine_similarity(track.embedding_template, candidate.embedding)
        identity_similarity = min(base_similarity, template_similarity)
        if identity_similarity < tracking_config.replacement_similarity:
            rejections["identity_similarity_too_low"] += 1
            continue
        if candidate.sharpness < base.sharpness * replacement_config.min_quality_ratio:
            rejections["sharpness_too_low"] += 1
            continue
        result.append((candidate, identity_similarity))
    result.sort(key=lambda item: _candidate_score(item[0], region), reverse=True)
    return result, dict(sorted(rejections.items()))


def _try_blend(
    target: np.ndarray,
    donor_image: np.ndarray,
    base: FaceObservation,
    donor: FaceObservation,
    region: str,
    config: ReplacementConfig,
) -> Tuple[Optional[np.ndarray], float]:
    if base.mesh is None or donor.mesh is None:
        return None, 0.0
    stable = ALIGN_FOR_EYES if region == "eyes" else ALIGN_FOR_MOUTH
    matrix, inliers = cv2.estimateAffinePartial2D(
        donor.mesh[stable], base.mesh[stable], method=cv2.LMEDS
    )
    if matrix is None:
        return None, 0.0
    x0, y0, x1, y1 = _roi_for_face(base, target.shape[:2])
    roi_width, roi_height = x1 - x0, y1 - y0
    local_matrix = matrix.copy()
    local_matrix[0, 2] -= x0
    local_matrix[1, 2] -= y0
    warped = cv2.warpAffine(
        donor_image,
        local_matrix,
        (roi_width, roi_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    target_roi = target[y0:y1, x0:x1]
    local_mesh = base.mesh - np.array([x0, y0], dtype=np.float32)
    feature_mask = _region_mask((roi_height, roi_width), local_mesh, region, base.size)

    local_base = FaceObservation(
        image_index=base.image_index,
        image_name=base.image_name,
        bbox=base.bbox - np.array([x0, y0, 0, 0], dtype=np.float32),
        landmarks5=base.landmarks5,
        detection_score=base.detection_score,
        embedding=base.embedding,
        sharpness=base.sharpness,
    )
    context = _face_context_mask((roi_height, roi_width), local_base)
    exclusion_radius = max(2, int(round(base.size * 0.08)))
    exclusion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * exclusion_radius + 1, 2 * exclusion_radius + 1)
    )
    outside = cv2.bitwise_and(
        context, cv2.bitwise_not(cv2.dilate(feature_mask, exclusion_kernel))
    )
    outside_similarity = _masked_ssim(target_roi, warped, outside)
    if outside_similarity < config.min_outside_similarity:
        return None, outside_similarity

    warped = _color_match(warped, target_roi, outside)
    sigma = max(1.0, base.size * config.feather_ratio)
    alpha = cv2.GaussianBlur(feature_mask.astype(np.float32) / 255.0, (0, 0), sigma)
    alpha = np.clip(alpha[:, :, None], 0.0, 1.0)
    blended = target_roi.astype(np.float32) * (1.0 - alpha) + warped.astype(np.float32) * alpha
    result = target.copy()
    result[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return result, outside_similarity


def perform_replacements(
    base_image: ImageAnalysis,
    base_pixels: np.ndarray,
    images: Sequence[ImageAnalysis],
    tracks: Sequence[PersonTrack],
    tracking_config: TrackingConfig,
    replacement_config: ReplacementConfig,
    manual: ManualConfig,
    image_loader,
) -> Tuple[np.ndarray, List[ReplacementRecord]]:
    output = base_pixels.copy()
    records: List[ReplacementRecord] = []
    loaded: Dict[int, np.ndarray] = {}
    replacement_count = 0
    excluded = set(manual.exclude_tracks)
    for track in tracks:
        base = track.observations.get(base_image.index)
        if base is None or track.track_id in excluded:
            continue
        regions = []
        if not base.eyes_good:
            regions.append("eyes")
        if replacement_config.replace_mouth and not base.mouth_good:
            regions.append("mouth")
        for region in regions:
            if replacement_count >= replacement_config.max_replacements:
                break
            record = ReplacementRecord(
                track_id=track.track_id,
                region=region,
                base_image=base_image.path.name,
                donor_image=None,
                status="skipped",
            )
            if base.mesh is None:
                record.reason = "base face landmarks unavailable"
                records.append(record)
                continue
            forced = manual.force_donors.get(track.track_id, {}).get(region)
            candidates, rejections = _candidate_observations(
                track, base, region, tracking_config, replacement_config, forced
            )
            record.candidate_rejections = rejections
            if not candidates:
                record.reason = "no candidate passed expression, quality and identity checks"
                records.append(record)
                continue
            attempted = []
            for donor, identity_similarity in candidates:
                if donor.image_index not in loaded:
                    loaded[donor.image_index] = image_loader(images[donor.image_index].path)
                proposal, outside_similarity = _try_blend(
                    output,
                    loaded[donor.image_index],
                    base,
                    donor,
                    region,
                    replacement_config,
                )
                attempted.append("{}:{:.3f}".format(donor.image_name, outside_similarity))
                if proposal is None:
                    continue
                output = proposal
                replacement_count += 1
                record.donor_image = donor.image_name
                record.status = "replaced"
                record.identity_similarity = identity_similarity
                record.outside_similarity = outside_similarity
                record.reason = "passed all conservative checks"
                break
            if record.status != "replaced":
                record.reason = "outside-face similarity too low ({})".format(", ".join(attempted))
            records.append(record)
    return output, records
