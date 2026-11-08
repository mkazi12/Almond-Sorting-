"""Model-boundary tests run with the Python standard library alone."""

import dataclasses
import unittest

from app.inference import (
    Detection, InferenceError, InferenceResult, RGBImage, validate_class_map, validate_result,
)


class InferenceContractTests(unittest.TestCase):
    def test_rgb_image_requires_exact_contiguous_bytes(self):
        self.assertEqual(RGBImage(2, 1, b"\x00" * 6).pixels, b"\x00" * 6)
        for width, height, pixels in ((0, 1, b""), (True, 1, b"xxx"), (1, 1, b"xx"), (1, 1, bytearray(3))):
            with self.subTest(width=width, height=height, pixels=pixels):
                with self.assertRaises(ValueError):
                    RGBImage(width, height, pixels)

    def test_detection_preserves_category_and_fractional_pixel_coordinates(self):
        detection = Detection(17, .92, [1.5, 2.25, 11.5, 12.25])
        self.assertEqual(detection.category_id, 17)
        self.assertEqual(detection.bbox_xyxy, (1.5, 2.25, 11.5, 12.25))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            detection.confidence = .1

    def test_confidence_must_be_finite_and_between_zero_and_one(self):
        for confidence in (-.1, 1.01, float("nan"), float("inf"), True, "0.5", 10 ** 1000):
            with self.subTest(confidence=confidence):
                with self.assertRaises(ValueError):
                    Detection(1, confidence, (1, 1, 2, 2))

    def test_boxes_require_finite_ordered_nonnegative_pixel_coordinates(self):
        cases = ((1, 1, 1, 2), (2, 1, 1, 2), (1, 2, 2, 1), (-1, 1, 2, 2),
                 (1, 1, float("inf"), 2), (1, 1, 2), (True, 1, 2, 2), None)
        for box in cases:
            with self.subTest(box=box):
                with self.assertRaises(ValueError):
                    Detection(1, .8, box)

    def test_category_ids_are_integers(self):
        for category in (-1, True, "1", 1.5):
            with self.subTest(category=category):
                with self.assertRaises(ValueError):
                    Detection(category, .8, (1, 1, 2, 2))

    def test_empty_result_is_a_success_with_recorded_floor(self):
        result = InferenceResult(20, 10, .25)
        self.assertEqual(result.detections, ())
        self.assertEqual(result.confidence_floor, .25)
        validate_result(result, image_width=20, image_height=10, confidence_floor=.25, classes={})

    def test_result_rejects_boxes_outside_original_dimensions(self):
        with self.assertRaisesRegex(ValueError, "dimensions"):
            InferenceResult(20, 10, .25, (Detection(1, .9, (0, 0, 21, 10)),))

    def test_result_rejects_predictions_below_recorded_floor(self):
        with self.assertRaisesRegex(ValueError, "floor"):
            InferenceResult(20, 10, .5, (Detection(1, .49, (0, 0, 20, 10)),))

    def test_result_copies_prediction_sequence(self):
        predictions = [Detection(1, .5, (0, 0, 20, 10))]
        result = InferenceResult(20, 10, .5, predictions)
        predictions.clear()
        self.assertEqual(len(result.detections), 1)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.confidence_floor = 0

    def test_wrong_image_or_floor_is_rejected_at_adapter_boundary(self):
        result = InferenceResult(20, 10, .25)
        for width, height, floor in ((40, 10, .25), (20, 20, .25), (20, 10, .5)):
            with self.subTest(width=width, height=height, floor=floor):
                with self.assertRaises(InferenceError):
                    validate_result(result, image_width=width, image_height=height,
                                    confidence_floor=floor, classes={})

    def test_unknown_or_blank_class_name_is_rejected(self):
        result = InferenceResult(20, 10, .25, (Detection(17, .9, (1, 1, 2, 2)),))
        for classes in ({1: "Almond"}, {17: " "}, {17: None}):
            with self.subTest(classes=classes):
                with self.assertRaisesRegex(InferenceError, "17"):
                    validate_result(result, image_width=20, image_height=10,
                                    confidence_floor=.25, classes=classes)

    def test_untyped_adapter_output_is_rejected(self):
        with self.assertRaisesRegex(InferenceError, "result type"):
            validate_result([], image_width=20, image_height=10, confidence_floor=.25, classes={})

    def test_configured_class_map_requires_integer_ids_and_nonblank_names(self):
        for classes in (None, {}, {"1": "Almond"}, {True: "Almond"}, {-1: "Almond"},
                        {1: " "}, {1: None}, {1: "Almond", 2: ""}):
            with self.subTest(classes=classes):
                with self.assertRaises(ValueError):
                    validate_class_map(classes)
        validate_class_map({17: "Almond", 28: "Foreign material"})


if __name__ == "__main__":
    unittest.main()
