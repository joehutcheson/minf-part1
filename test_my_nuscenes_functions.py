import math
from unittest import TestCase
import numpy as np

from my_nuscenes_functions import find_translation, rotation, find_dist_between_ranges


# TODO: Review all tests in this file

class TestFindTranslation(TestCase):
    # TODO: Rewrite with updated functionality

    def test_1(self):
        # Two boxes placed beside each other. 
        ego_bb = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        ann_bb = np.array([[2, 0], [3, 0], [3, 1], [2, 1]])
        expected = [1, 0]

        for _ in range(4):
            actual = find_translation(ego_bb, ann_bb)
            np.testing.assert_allclose(expected, actual)
            ego_bb = np.roll(ego_bb, 1, axis=0)

        for _ in range(4):
            actual = find_translation(ego_bb, ann_bb)
            np.testing.assert_allclose(expected, actual)
            ann_bb = np.roll(ann_bb, 1, axis=0)

    def test_2(self):
        # Two boxes placed beside each other
        ego_bb = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        ann_bb = np.array([[0, 2], [1, 2], [1, 3], [0, 3]])
        expected = [0, 1]

        for _ in range(4):
            actual = find_translation(ego_bb, ann_bb)
            np.testing.assert_allclose(expected, actual)
            ego_bb = np.roll(ego_bb, 1, axis=0)

        for _ in range(4):
            actual = find_translation(ego_bb, ann_bb)
            np.testing.assert_allclose(expected, actual)
            ann_bb = np.roll(ann_bb, 1, axis=0)

    def test_3(self):
        # Two boxes placed diagonally from each other
        ego_bb = np.array([[3, 3], [4, 3], [4, 4], [3, 4]])
        ann_bb = np.array([[1, 1], [2, 1], [2, 2], [1, 2]])
        expected = [-1, -1]

        for _ in range(4):
            actual = find_translation(ego_bb, ann_bb)
            np.testing.assert_allclose(expected, actual)
            ego_bb = np.roll(ego_bb, 1, axis=0)

        for _ in range(4):
            actual = find_translation(ego_bb, ann_bb)
            np.testing.assert_allclose(expected, actual)
            ann_bb = np.roll(ann_bb, 1, axis=0)


class TestFindDistBetweenRanges(TestCase):
    def test_behind(self):
        range_1 = [0, 5]
        range_2 = [10, 15]

        actual = find_dist_between_ranges(range_1, range_2)
        expected = 5

        self.assertAlmostEqual(actual, expected)

    def test_inline(self):
        range_1 = [0, 5]
        range_2 = [1, 4]

        actual = find_dist_between_ranges(range_1, range_2)
        expected = 0

        self.assertAlmostEqual(actual, expected)

    def test_inline_2(self):
        range_1 = [5, 0]
        range_2 = [0, 5]

        actual = find_dist_between_ranges(range_1, range_2)
        expected = 0

        self.assertAlmostEqual(actual, expected)

    def test_inline_back(self):
        range_1 = [0, 5]
        range_2 = [5, 15]

        actual = find_dist_between_ranges(range_1, range_2)
        expected = 0

        self.assertAlmostEqual(actual, expected)

    def test_inline_front(self):
        range_1 = [10, 15]
        range_2 = [5, 10]

        actual = find_dist_between_ranges(range_1, range_2)
        expected = 0

        self.assertAlmostEqual(actual, expected)

    def test_ahead(self):
        range_1 = [18, 21]
        range_2 = [10, 15]

        actual = find_dist_between_ranges(range_1, range_2)
        expected = -3

        self.assertAlmostEqual(actual, expected)


class TestRotation(TestCase):
    def test_90_degree_1(self):
        angle = np.pi / 2
        vector = [0, 1]
        expected = np.array([-1, 0])
        actual = rotation(angle, vector)
        np.testing.assert_allclose(expected, actual, atol=0.00001)

    def test_90_degree_2(self):
        angle = np.pi / 2
        vector = [1, 0]
        expected = np.array([0, 1])
        actual = rotation(angle, vector)
        np.testing.assert_allclose(expected, actual, atol=0.00001)

    def test_90_degree_3(self):
        angle = np.pi / 2
        vector = [0, -1]
        expected = np.array([1, 0])
        actual = rotation(angle, vector)
        np.testing.assert_allclose(expected, actual, atol=0.00001)


    def test_90_degree_4(self):
        angle = np.pi / 2
        vector = [-1, 0]
        expected = np.array([0, -1])
        actual = rotation(angle, vector)
        np.testing.assert_allclose(expected, actual, atol=0.00001)

    def test_neg_90_degree(self):
        angle = -np.pi / 2
        vector = [1, 0]
        expected = np.array([0, -1])
        actual = rotation(angle, vector)
        np.testing.assert_allclose(expected, actual, atol=0.00001)

    def test_45_degree(self):
        angle = np.pi / 4
        vector = [1, 0]
        expected = np.array([1/math.sqrt(2), 1/math.sqrt(2)])
        actual = rotation(angle, vector)
        np.testing.assert_allclose(expected, actual, atol=0.00001)
