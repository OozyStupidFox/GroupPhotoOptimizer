from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import TrackingConfig
from .detection import cosine_similarity
from .types import FaceObservation, ImageAnalysis, PersonTrack


def _normalize(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


def _refresh_track(track: PersonTrack) -> None:
    observations = list(track.observations.values())
    track.canonical_center = np.median(
        np.stack([item.canonical_center for item in observations]), axis=0
    ).astype(np.float32)
    track.canonical_size = float(np.median([item.canonical_size for item in observations]))
    track.embedding_template = _normalize(
        np.mean(np.stack([item.embedding for item in observations]), axis=0)
    ).astype(np.float32)


def _track_sort_key(track: PersonTrack) -> Tuple[float, float]:
    return float(track.canonical_center[1]), float(track.canonical_center[0])


def _match_image(
    tracks: List[PersonTrack],
    observations: List[FaceObservation],
    config: TrackingConfig,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    if not tracks or not observations:
        return [], list(range(len(observations)))
    impossible = 1e6
    cost = np.full((len(tracks), len(observations)), impossible, dtype=np.float64)
    for track_index, track in enumerate(tracks):
        for obs_index, obs in enumerate(observations):
            size = max((track.canonical_size + obs.canonical_size) / 2.0, 1.0)
            position = float(np.linalg.norm(track.canonical_center - obs.canonical_center) / size)
            similarity = cosine_similarity(track.embedding_template, obs.embedding)
            if position > config.max_position_in_face_widths:
                continue
            # Extremely close positions tolerate a slightly noisy tiny-face embedding.
            minimum_similarity = config.match_similarity if position > 0.34 else config.match_similarity - 0.10
            if similarity < minimum_similarity:
                continue
            cost[track_index, obs_index] = 0.72 * position + 1.65 * (1.0 - similarity)
    row_indices, column_indices = linear_sum_assignment(cost)
    matched: List[Tuple[int, int]] = []
    used_observations = set()
    for track_index, obs_index in zip(row_indices, column_indices):
        if cost[track_index, obs_index] < impossible:
            matched.append((int(track_index), int(obs_index)))
            used_observations.add(int(obs_index))
    unmatched = [index for index in range(len(observations)) if index not in used_observations]
    return matched, unmatched


def _supplemental_tracks(
    unmatched: Dict[int, List[FaceObservation]], config: TrackingConfig
) -> List[PersonTrack]:
    pool = [item for observations in unmatched.values() for item in observations]
    pool.sort(key=lambda item: item.detection_score, reverse=True)
    consumed = set()
    result: List[PersonTrack] = []
    for seed in pool:
        if id(seed) in consumed:
            continue
        candidates: Dict[int, Tuple[float, FaceObservation]] = {}
        for other in pool:
            if id(other) in consumed or other.image_index == seed.image_index:
                continue
            size = max((seed.canonical_size + other.canonical_size) / 2.0, 1.0)
            position = float(np.linalg.norm(seed.canonical_center - other.canonical_center) / size)
            similarity = cosine_similarity(seed.embedding, other.embedding)
            if position <= 0.72 and similarity >= config.match_similarity:
                score = position + 1.5 * (1.0 - similarity)
                previous = candidates.get(other.image_index)
                if previous is None or score < previous[0]:
                    candidates[other.image_index] = (score, other)
        cluster = [seed] + [value[1] for value in candidates.values()]
        if len(cluster) < config.min_observations_for_new_track:
            continue
        track = PersonTrack(track_id="pending")
        for item in cluster:
            track.add(item)
            consumed.add(id(item))
        _refresh_track(track)
        result.append(track)
    return result


def build_tracks(images: Sequence[ImageAnalysis], config: TrackingConfig) -> List[PersonTrack]:
    if not images:
        return []
    reference = max(images, key=lambda item: len(item.observations))
    tracks: List[PersonTrack] = []
    for observation in reference.observations:
        track = PersonTrack(track_id="pending")
        track.add(observation)
        _refresh_track(track)
        tracks.append(track)

    unmatched: Dict[int, List[FaceObservation]] = {}
    order = sorted(
        (item for item in images if item.index != reference.index),
        key=lambda item: len(item.observations),
        reverse=True,
    )
    for image in order:
        matched, remaining = _match_image(tracks, image.observations, config)
        for track_index, obs_index in matched:
            tracks[track_index].add(image.observations[obs_index])
            _refresh_track(tracks[track_index])
        unmatched[image.index] = [image.observations[index] for index in remaining]

    tracks.extend(_supplemental_tracks(unmatched, config))
    tracks.sort(key=_track_sort_key)
    for index, track in enumerate(tracks, start=1):
        track.track_id = "P{:03d}".format(index)
        for observation in track.observations.values():
            observation.track_id = track.track_id
        _refresh_track(track)
    return tracks


def assign_canonical_geometry(images: Iterable[ImageAnalysis], transform_points, transformed_size) -> None:
    for image in images:
        for observation in image.observations:
            observation.canonical_center = transform_points(
                observation.center.reshape(1, 2), image.homography_to_canonical
            )[0]
            observation.canonical_size = transformed_size(
                observation.bbox, image.homography_to_canonical
            )

