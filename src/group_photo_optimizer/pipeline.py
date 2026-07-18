from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from .alignment import estimate_homography, transform_points, transformed_face_size
from .blending import perform_replacements
from .config import AppConfig, ResolvedPaths, to_plain_dict
from .detection import FaceDetector
from .models import model_fingerprint, resolve_models
from .report import (
    create_annotated_base,
    create_replacement_contact_sheet,
    write_audit_json,
    write_html_report,
    write_observation_csv,
)
from .scoring import choose_base_image, image_score, score_tracks
from .runlog import RunLog, create_run_log
from .tracking import assign_canonical_geometry, build_tracks
from .types import ImageAnalysis, PersonTrack


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class PipelineResult:
    output_dir: Path
    final_path: Path
    report_path: Path
    log_path: Path
    total_seconds: float
    base_image: str
    replaced_count: int
    skipped_count: int


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Could not read image: {}".format(path))
    return image


def save_final_jpeg(path: Path, bgr: np.ndarray, metadata_source: Path) -> None:
    source = Image.open(metadata_source)
    exif = source.info.get("exif", b"")
    icc_profile = source.info.get("icc_profile")
    source.close()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(
        path,
        format="JPEG",
        quality=98,
        subsampling=0,
        exif=exif,
        icc_profile=icc_profile,
    )


def _cache_key(
    config: AppConfig, image_paths: Sequence[Path], fingerprint: str
) -> str:
    analysis_config = {
        "detection": to_plain_dict(config.detection),
        "alignment": to_plain_dict(config.alignment),
        "tracking": to_plain_dict(config.tracking),
        "scoring": to_plain_dict(config.scoring),
    }
    payload = {
        "version": 2,
        "config": analysis_config,
        "models": fingerprint,
        "images": [
            [path.name, path.stat().st_size, path.stat().st_mtime_ns] for path in image_paths
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_cache(path: Path, key: str) -> Tuple[List[ImageAnalysis], List[PersonTrack]]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if data.get("key") != key:
        raise ValueError("cache does not match images, models or analysis configuration")
    return data["images"], data["tracks"]


def _save_cache(
    path: Path, key: str, images: List[ImageAnalysis], tracks: List[PersonTrack]
) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump({"key": key, "images": images, "tracks": tracks}, handle, protocol=4)
    temporary.replace(path)


def analyze_photos(
    config: AppConfig,
    paths: ResolvedPaths,
    image_paths: Sequence[Path],
    model_paths,
    run_log: RunLog,
) -> Tuple[List[ImageAnalysis], List[PersonTrack]]:
    analyses: List[ImageAnalysis] = []
    stage_started = time.perf_counter()
    detector = FaceDetector(
        model_paths["yunet"], model_paths["sface"], model_paths["face_landmarker"], config.detection
    )
    try:
        for index, path in enumerate(tqdm(image_paths, desc="Detecting and analyzing faces")):
            image_started = time.perf_counter()
            pixels = read_image(path)
            observations = detector.analyze(pixels, index, path.name)
            analyses.append(
                ImageAnalysis(
                    index=index,
                    path=path,
                    width=pixels.shape[1],
                    height=pixels.shape[0],
                    observations=observations,
                )
            )
            run_log.logger.info(
                "DETECTION image=%s faces=%d landmarks=%d seconds=%.3f",
                path.name,
                len(observations),
                sum(item.mesh is not None for item in observations),
                time.perf_counter() - image_started,
            )
    finally:
        detector.close()
    run_log.stage("face_detection_and_expression", stage_started)

    stage_started = time.perf_counter()
    reference = max(analyses, key=lambda item: len(item.observations))
    run_log.logger.info("ALIGNMENT reference_image=%s", reference.path.name)
    reference_pixels = read_image(reference.path)
    for analysis in tqdm(analyses, desc="Registering photos"):
        if analysis.index == reference.index:
            analysis.homography_to_canonical = np.eye(3, dtype=np.float64)
            analysis.alignment_inliers = 9999
            analysis.alignment_ok = True
            run_log.logger.info(
                "ALIGNMENT image=%s inliers=%d ok=true seconds=0.000",
                analysis.path.name,
                analysis.alignment_inliers,
            )
            continue
        image_started = time.perf_counter()
        homography, inliers, ok = estimate_homography(
            read_image(analysis.path), reference_pixels, config.alignment
        )
        analysis.homography_to_canonical = homography
        analysis.alignment_inliers = inliers
        analysis.alignment_ok = ok
        run_log.logger.info(
            "ALIGNMENT image=%s inliers=%d ok=%s seconds=%.3f",
            analysis.path.name,
            inliers,
            str(ok).lower(),
            time.perf_counter() - image_started,
        )
        if not ok:
            raise RuntimeError(
                "Photo registration failed for {} ({} inliers); remove the photo or lower alignment.min_matches"
                .format(analysis.path.name, inliers)
            )
    run_log.stage("photo_registration", stage_started)

    stage_started = time.perf_counter()
    assign_canonical_geometry(analyses, transform_points, transformed_face_size)
    tracks = build_tracks(analyses, config.tracking)
    score_tracks(tracks, config.scoring)
    run_log.logger.info(
        "TRACKING tracks=%d full_length_tracks=%d",
        len(tracks),
        sum(len(track.observations) == len(analyses) for track in tracks),
    )
    run_log.stage("tracking_and_scoring", stage_started)
    return analyses, tracks


def _log_quality_ranking(run_log: RunLog, images: Sequence[ImageAnalysis]) -> None:
    run_log.logger.info("QUALITY_RANKING_BEGIN count=%d", len(images))
    for rank, image in enumerate(sorted(images, key=image_score, reverse=True), start=1):
        good, open_eyes, analyzed, quality_sum = image_score(image)
        run_log.logger.info(
            "QUALITY_RANK rank=%d image=%s good_faces=%d open_eye_faces=%d "
            "landmark_faces=%d detected_faces=%d quality_sum=%.6f selection_key=%s",
            rank,
            image.path.name,
            good,
            open_eyes,
            analyzed,
            len(image.observations),
            quality_sum,
            (good, open_eyes, analyzed, round(quality_sum, 6)),
        )
    run_log.logger.info("QUALITY_RANKING_END")


def _log_replacements(
    run_log: RunLog,
    base: ImageAnalysis,
    tracks: Sequence[PersonTrack],
    records,
) -> None:
    tracks_by_id = {track.track_id: track for track in tracks}
    run_log.logger.info("REPLACEMENT_LIST_BEGIN count=%d", len(records))
    for record in records:
        track = tracks_by_id.get(record.track_id)
        base_observation = None if track is None else track.observations.get(base.index)
        base_blink = -1.0 if base_observation is None else base_observation.eye_blink
        base_mouth = -1.0 if base_observation is None else base_observation.mouth_abnormal
        observations = 0 if track is None else len(track.observations)
        if record.status == "replaced":
            run_log.logger.info(
                "REPLACED track=%s region=%s donor=%s identity_similarity=%.6f "
                "outside_similarity=%.6f base_blink=%.6f base_mouth=%.6f",
                record.track_id,
                record.region,
                record.donor_image,
                record.identity_similarity,
                record.outside_similarity,
                base_blink,
                base_mouth,
            )
        else:
            run_log.logger.info(
                "NO_REPLACEMENT track=%s region=%s observations=%d base_blink=%.6f "
                "base_mouth=%.6f reason=%s candidate_rejections=%s",
                record.track_id,
                record.region,
                observations,
                base_blink,
                base_mouth,
                record.reason,
                json.dumps(record.candidate_rejections, ensure_ascii=False, sort_keys=True),
            )
    run_log.logger.info(
        "REPLACEMENT_SUMMARY replaced=%d no_replacement=%d",
        sum(record.status == "replaced" for record in records),
        sum(record.status != "replaced" for record in records),
    )
    run_log.logger.info("REPLACEMENT_LIST_END")


def run_pipeline(
    config_path: Path,
    reuse_analysis: bool = False,
    analyze_only: bool = False,
    input_dir_override: Optional[Path] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> PipelineResult:
    config_path = config_path.resolve()
    config = AppConfig.load(config_path)
    paths = config.resolve_paths(config_path)
    if input_dir_override is not None:
        paths = ResolvedPaths(
            root=paths.root,
            input_dir=input_dir_override.resolve(),
            output_dir=paths.output_dir,
            models_dir=paths.models_dir,
        )
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    run_log = create_run_log(paths.output_dir, callback=log_callback)
    run_log.logger.info(
        "RUN started config=%s input_dir=%s output_dir=%s reuse_analysis=%s analyze_only=%s",
        config_path,
        paths.input_dir,
        paths.output_dir,
        reuse_analysis,
        analyze_only,
    )
    try:
        stage_started = time.perf_counter()
        model_paths = resolve_models(paths.models_dir)
        run_log.logger.info(
            "MODELS %s",
            " ".join("{}={}".format(name, path) for name, path in sorted(model_paths.items())),
        )
        run_log.stage("model_resolution", stage_started)

        image_paths = sorted(
            path for path in paths.input_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
        )
        run_log.logger.info("INPUT images=%d", len(image_paths))
        if len(image_paths) < 2:
            raise ValueError("At least two images are required in {}".format(paths.input_dir))
        key = _cache_key(config, image_paths, model_fingerprint(model_paths))
        cache_path = paths.output_dir / "analysis_cache.pkl"
        stage_started = time.perf_counter()
        if reuse_analysis and cache_path.exists():
            run_log.logger.info("CACHE reuse path=%s", cache_path)
            try:
                images, tracks = _load_cache(cache_path, key)
            except ValueError as error:
                run_log.logger.warning("CACHE invalid reason=%s; running fresh analysis", error)
                images, tracks = analyze_photos(
                    config, paths, image_paths, model_paths, run_log
                )
                _save_cache(cache_path, key, images, tracks)
        else:
            run_log.logger.info("CACHE fresh_analysis path=%s", cache_path)
            images, tracks = analyze_photos(
                config, paths, image_paths, model_paths, run_log
            )
            _save_cache(cache_path, key, images, tracks)
        run_log.stage("analysis_or_cache", stage_started)

        _log_quality_ranking(run_log, images)
        base = choose_base_image(images, config.manual)
        base_pixels = read_image(base.path)
        run_log.logger.info(
            "BASE_SELECTED image=%s score=%s tracks=%d detected_faces=%d",
            base.path.name,
            image_score(base),
            len(tracks),
            len(base.observations),
        )
        create_annotated_base(base_pixels, base, paths.output_dir / "base_annotated.jpg")

        stage_started = time.perf_counter()
        if analyze_only:
            records = []
            final_pixels = base_pixels
            run_log.logger.info("REPLACEMENT skipped because analyze_only=true")
        else:
            final_pixels, records = perform_replacements(
                base, base_pixels, images, tracks, config.tracking, config.replacement,
                config.manual, read_image,
            )
        run_log.stage("feature_replacement", stage_started)
        _log_replacements(run_log, base, tracks, records)

        stage_started = time.perf_counter()
        save_final_jpeg(paths.output_dir / "final.jpg", final_pixels, base.path)
        has_contact_sheet = create_replacement_contact_sheet(
            base_pixels,
            final_pixels,
            base,
            images,
            tracks,
            records,
            paths.output_dir / "replacement_review.jpg",
            read_image,
        )
        write_observation_csv(paths.output_dir / "observations.csv", images)
        write_audit_json(paths.output_dir / "audit.json", images, tracks, base, records)
        write_html_report(
            paths.output_dir / "report.html", images, tracks, base, records, has_contact_sheet
        )
        run_log.stage("output_writing", stage_started)
        run_log.logger.info("OUTPUT final=%s", paths.output_dir / "final.jpg")
        run_log.logger.info("OUTPUT review=%s", paths.output_dir / "report.html")
        total_seconds = run_log.finish(paths.output_dir)
        return PipelineResult(
            output_dir=paths.output_dir,
            final_path=paths.output_dir / "final.jpg",
            report_path=paths.output_dir / "report.html",
            log_path=run_log.path,
            total_seconds=total_seconds,
            base_image=base.path.name,
            replaced_count=sum(record.status == "replaced" for record in records),
            skipped_count=sum(record.status != "replaced" for record in records),
        )
    except Exception:
        run_log.logger.exception(
            "RUN failed total_seconds=%.3f", time.perf_counter() - run_log.started_at
        )
        run_log.fail(paths.output_dir)
        raise
