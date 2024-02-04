from unittest import TestCase
import numpy as np

from scoring import find_min_long_distance, find_min_long_distance_opposite_direction, \
    find_min_lat_distance, generate_individual_score, is_right_of, angle_to_vector
from constants import *


class TestFindMinLongDistance(TestCase):
    """
    These tests use results which were calculated by hand using lemma 2 from On a Formal Model of Safe and Scalable
    Self-driving Cars.
    """

    def test_1(self):
        v_r = 3
        v_f = 5

        expected = 40.9583
        actual = find_min_long_distance(v_r, v_f, rss_conservative)

        self.assertAlmostEqual(expected, actual, places=3)

    def test_2(self):
        v_r = 5
        v_f = 3

        expected = 8.2098
        actual = find_min_long_distance(v_r, v_f, rss_aggressive)

        self.assertAlmostEqual(expected, actual, places=3)

    def test_3(self):
        v_r = 0
        v_f = 5

        expected = 0
        actual = find_min_long_distance(v_r, v_f, rss_aggressive)

        self.assertAlmostEqual(expected, actual, places=3)

    def test_4(self):
        v_r = -1
        v_f = 1

        with self.assertRaises(AssertionError):
            find_min_long_distance(v_r, v_f, rss_aggressive)

    def test_5(self):
        v_r = 1
        v_f = -1

        with self.assertRaises(AssertionError):
            find_min_long_distance(v_r, v_f, rss_conservative)


class TestFindMinLongDistanceOppositeDirection(TestCase):
    """
    These tests use results which were calculated by hand using lemma 3 from On a Formal Model of Safe and Scalable
    Self-driving Cars.
    """

    def test_1(self):
        v_1 = 3
        v_2 = -2

        expected = 110.1434
        actual = find_min_long_distance_opposite_direction(v_1, v_2, rss_conservative)

        self.assertAlmostEqual(expected, actual, places=3)

    def test_2(self):
        v_1 = 2
        v_2 = -3

        expected = 11.6324
        actual = find_min_long_distance_opposite_direction(v_1, v_2, rss_aggressive)

        self.assertAlmostEqual(expected, actual, places=3)

    def test_3(self):
        v_1 = -2
        v_2 = -3

        with self.assertRaises(AssertionError):
            find_min_long_distance_opposite_direction(v_1, v_2, rss_conservative)

    def test_4(self):
        v_1 = 2
        v_2 = 3

        with self.assertRaises(AssertionError):
            find_min_long_distance_opposite_direction(v_1, v_2, rss_conservative)


class TestFindMinLatDistance(TestCase):
    """
    These tests use results which were calculated by hand using lemma 4 from On a Formal Model of Safe and Scalable
    Self-driving Cars.
    """

    def test_1(self):
        v_1 = 2.5
        v_2 = -1

        expected = 17.2078
        actual = find_min_lat_distance(v_1, v_2, rss_conservative)

        self.assertAlmostEqual(expected, actual, places=3)

    def test_2(self):
        v_1 = 0.5
        v_2 = -1.5

        expected = 3.0817
        actual = find_min_lat_distance(v_1, v_2, rss_aggressive)

        self.assertAlmostEqual(expected, actual, places=3)

    def test_3(self):
        v_1 = -0.1
        v_2 = -0.1

        expected = rss_conservative['mu']
        actual = find_min_lat_distance(v_1, v_2, rss_conservative)

        self.assertAlmostEqual(expected, actual, places=3)

    def test_4(self):
        v_1 = 0.01
        v_2 = 0.01

        expected = rss_aggressive['mu']
        actual = find_min_lat_distance(v_1, v_2, rss_aggressive)

        self.assertAlmostEqual(expected, actual, places=3)


class TestGenerateIndividualScore(TestCase):
    def test_1(self):
        minimum = 5
        actual = 5

        expected = 0
        result = generate_individual_score(minimum, actual)

        self.assertAlmostEqual(expected, result)

    def test_2(self):
        minimum = 1
        actual = 6

        expected = 1
        result = generate_individual_score(minimum, actual)

        self.assertAlmostEqual(expected, result)

    def test_3(self):
        minimum = 50
        actual = 5

        expected = 0
        result = generate_individual_score(minimum, actual)

        self.assertAlmostEqual(expected, result)

    def test_4(self):
        minimum = 10
        actual = 12

        expected = 0.4
        result = generate_individual_score(minimum, actual)

        self.assertAlmostEqual(expected, result)

    def test_5(self):
        minimum = 14
        actual = 14.5

        expected = 0.5
        result = generate_individual_score(minimum, actual, gradient=1)

        self.assertAlmostEqual(expected, result)


class TestIsRightOf(TestCase):
    def test_0_degree_true(self):
        theta = 0
        p1 = [1, 0]
        p2 = [-1, 0]
        self.assertTrue(is_right_of(theta, p1, p2))

    def test_0_degree_false(self):
        theta = 0
        p1 = [-1, 0]
        p2 = [1, 0]
        self.assertFalse(is_right_of(theta, p1, p2))

    def test_0_degree_inline(self):
        theta = 0
        p1 = [0, 0]
        p2 = [0, -1]

        self.assertIsNone(is_right_of(theta, p1, p2))

    def test_90_degree_true(self):
        theta = np.pi/2
        p1 = [0, -1]
        p2 = [0, 1]
        self.assertTrue(is_right_of(theta, p1, p2))

    def test_90_degree_false(self):
        theta = np.pi/2
        p1 = [0, 1]
        p2 = [0, -1]

        self.assertFalse(is_right_of(theta, p1, p2))

    def test_180_degree_true(self):
        theta = np.pi
        p1 = [-1, 0]
        p2 = [1, 0]
        self.assertTrue(is_right_of(theta, p1, p2))

    def test_180_degree_false(self):
        theta = np.pi
        p1 = [1, 0]
        p2 = [-1, 0]
        self.assertFalse(is_right_of(theta, p1, p2))

    def test_270_degree_true(self):
        theta = 3 * np.pi/2
        p1 = [0, 1]
        p2 = [0, -1]
        self.assertTrue(is_right_of(theta, p1, p2))

    def test_270_degree_false(self):
        theta = 3 * np.pi/2
        p1 = [0, -1]
        p2 = [0, 1]

        self.assertFalse(is_right_of(theta, p1, p2))


class TestAngleToVector(TestCase):
    def test_1(self):
        a = 0

        expected = [0, 1]
        actual = angle_to_vector(a)

        np.testing.assert_allclose(expected, actual, atol=0.00001)

    def test_2(self):
        a = np.pi / 2

        expected = [-1, 0]
        actual = angle_to_vector(a)

        np.testing.assert_allclose(expected, actual, atol=0.00001)

    def test_3(self):
        a = np.pi

        expected = [0, -1]
        actual = angle_to_vector(a)

        np.testing.assert_allclose(expected, actual, atol=0.00001)

    def test_4(self):
        a = 3 * np.pi / 2

        expected = [1, 0]
        actual = angle_to_vector(a)

        np.testing.assert_allclose(expected, actual, atol=0.00001)

    def test_5(self):
        a = 2 * np.pi

        expected = [0, 1]
        actual = angle_to_vector(a)

        np.testing.assert_allclose(expected, actual, atol=0.00001)

    def test_6(self):
        a = np.pi / 4

        expected = [-1 / np.sqrt(2), 1 / np.sqrt(2)]
        actual = angle_to_vector(a)

        np.testing.assert_allclose(expected, actual, atol=0.00001)
