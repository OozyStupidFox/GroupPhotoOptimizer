from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .config import AlignmentConfig


def _work_image(image: np.ndarray, work_width: int) -> Tuple[np.ndarray, float]:
    scale = min(1.0, work_width / float(image.shape[1]))
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return gray, scale


def estimate_homography(
    source: np.ndarray, target: np.ndarray, config: AlignmentConfig
) -> Tuple[np.ndarray, int, bool]:
    source_gray, source_scale = _work_image(source, config.work_width)
    target_gray, target_scale = _work_image(target, config.work_width)
    orb = cv2.ORB_create(nfeatures=9000, scaleFactor=1.2, nlevels=8, fastThreshold=12)
    source_kp, source_desc = orb.detectAndCompute(source_gray, None)
    target_kp, target_desc = orb.detectAndCompute(target_gray, None)
    if source_desc is None or target_desc is None:
        return np.eye(3, dtype=np.float64), 0, False
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(source_desc, target_desc, k=2)
    good = [first for first, second in pairs if first.distance < 0.72 * second.distance]
    if len(good) < config.min_matches:
        return np.eye(3, dtype=np.float64), len(good), False
    source_points = np.float32([source_kp[item.queryIdx].pt for item in good]) / source_scale
    target_points = np.float32([target_kp[item.trainIdx].pt for item in good]) / target_scale
    homography, mask = cv2.findHomography(
        source_points, target_points, cv2.RANSAC, config.ransac_threshold
    )
    if homography is None or mask is None:
        return np.eye(3, dtype=np.float64), 0, False
    inliers = int(mask.sum())
    ok = inliers >= config.min_matches
    return homography.astype(np.float64), inliers, ok


def transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.copy()
    return cv2.perspectiveTransform(points.reshape(1, -1, 2).astype(np.float32), homography)[0]


def transformed_face_size(bbox: np.ndarray, homography: np.ndarray) -> float:
    x, y, w, h = bbox
    points = np.array([[x, y], [x + w, y], [x, y + h]], dtype=np.float32)
    mapped = transform_points(points, homography)
    return float((np.linalg.norm(mapped[1] - mapped[0]) + np.linalg.norm(mapped[2] - mapped[0])) / 2.0)
