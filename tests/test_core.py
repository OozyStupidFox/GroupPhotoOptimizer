import unittest
from pathlib import Path

import numpy as np

from group_photo_optimizer.alignment import transform_points
from group_photo_optimizer.config import ScoringConfig, TrackingConfig
from group_photo_optimizer.detection import _global_nms, _tile_starts
from group_photo_optimizer.scoring import score_tracks
from group_photo_optimizer.tracking import build_tracks
from group_photo_optimizer.types import FaceObservation, ImageAnalysis, PersonTrack


def observation(image_index, center, embedding, blink=0.0, mouth=0.0):
    x, y = center
    blendshapes = {
        "eyeBlinkLeft": blink,
        "eyeBlinkRight": blink,
        "mouthPucker": mouth,
    }
    return FaceObservation(
        image_index=image_index,
        image_name="{}.jpg".format(image_index),
        bbox=np.array([x - 20, y - 20, 40, 40], dtype=np.float32),
        landmarks5=np.zeros((5, 2), dtype=np.float32),
        detection_score=0.95,
        embedding=np.asarray(embedding, dtype=np.float32),
        sharpness=120.0,
        mesh=np.zeros((478, 2), dtype=np.float32),
        blendshapes=blendshapes,
        canonical_center=np.array(center, dtype=np.float32),
        canonical_size=40.0,
    )


class DetectionUtilitiesTest(unittest.TestCase):
    def test_tiles_cover_last_pixel(self):
        starts = _tile_starts(6240, 1600, 0.30)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1] + 1600, 6240)

    def test_global_nms_removes_tile_duplicates(self):
        first = np.array([10, 10, 40, 40] + [0] * 10 + [0.95], dtype=np.float32)
        duplicate = np.array([11, 11, 40, 40] + [0] * 10 + [0.90], dtype=np.float32)
        separate = np.array([100, 100, 40, 40] + [0] * 10 + [0.85], dtype=np.float32)
        kept = _global_nms([first, duplicate, separate], 0.3)
        self.assertEqual(len(kept), 2)
        self.assertAlmostEqual(float(kept[0][-1]), 0.95, places=4)


class TrackingAndScoringTest(unittest.TestCase):
    def test_position_and_embedding_keep_similar_people_separate(self):
        # Embeddings are deliberately similar; stable positions must preserve identity.
        images = []
        for image_index, shift in enumerate([0, 2, -1]):
            observations = [
                observation(image_index, (100 + shift, 100), [1.0, 0.10]),
                observation(image_index, (180 + shift, 100), [0.96, 0.28]),
            ]
            images.append(ImageAnalysis(image_index, Path("{}.jpg".format(image_index)), 400, 300, observations))
        tracks = build_tracks(images, TrackingConfig())
        self.assertEqual(len(tracks), 2)
        self.assertTrue(all(len(track.observations) == 3 for track in tracks))
        self.assertLess(tracks[0].canonical_center[0], tracks[1].canonical_center[0])

    def test_blink_is_not_good(self):
        track = PersonTrack("P001")
        open_face = observation(0, (100, 100), [1.0, 0.0], blink=0.05)
        closed_face = observation(1, (100, 100), [1.0, 0.0], blink=0.91)
        track.add(open_face)
        track.add(closed_face)
        score_tracks([track], ScoringConfig())
        self.assertTrue(open_face.eyes_good)
        self.assertFalse(closed_face.eyes_good)

    def test_perspective_point_transform(self):
        points = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        matrix = np.array([[1, 0, 10], [0, 1, 20], [0, 0, 1]], dtype=np.float64)
        transformed = transform_points(points, matrix)
        np.testing.assert_allclose(transformed, points + [10, 20])


if __name__ == "__main__":
    unittest.main()
