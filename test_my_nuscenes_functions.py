import math
from unittest import TestCase
import numpy as np

from my_nuscenes_functions import find_translation, rotation, find_dist_between_ranges


def test_with_rotation(ego_bb, ann_bb, expected):
    back = np.array(ego_bb[-1])
    front = np.array(ego_bb[0])

    ego_heading = front - back

    ego_heading = np.arctan2(ego_heading[0], ego_heading[1])

    for _ in range(4):
        actual = find_translation(ego_bb, ann_bb, ego_heading)
        np.testing.assert_allclose(expected, actual, atol=0.00001)
        ann_bb = np.roll(ann_bb, 1, axis=0)


class TestFindTranslation(TestCase):
    # TODO: Rewrite with updated functionality

    def test_1(self):
        ego_bb = np.array(([1, 1], [0, 1], [0, 0], [1, 0]))
        ann_bb = np.array([[3, 1], [2, 1], [2, 0], [3, 0]])
        expected = [1, 0]

        test_with_rotation(ego_bb, ann_bb, expected)

    def test_2(self):
        ego_bb = np.array([[3, 3], [2, 3], [2, 1], [3, 1]])
        ann_bb = np.array([[4, 3], [6, 4], [5, 6], [3, 5]])
        expected = [0, 0]

        test_with_rotation(ego_bb, ann_bb, expected)

    def test_3(self):
        ego_bb = np.array([[6, 3], [6, 4], [4, 4], [4, 3]])
        ann_bb = np.array([[2, 4], [1, 4], [1, 2], [2, 2]])
        expected = [-2, 0]

        test_with_rotation(ego_bb, ann_bb, expected)

    def test_4(self):
        ego_bb = np.array([[5, 6], [4, 6], [4, 4], [5, 4]])
        ann_bb = np.array([[2, 1], [1, 2], [0, 1], [1, 0]])
        expected = [-2, -2]

        test_with_rotation(ego_bb, ann_bb, expected)


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
        expected = np.array([1 / math.sqrt(2), 1 / math.sqrt(2)])
        actual = rotation(angle, vector)
        np.testing.assert_allclose(expected, actual, atol=0.00001)
