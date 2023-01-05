from unittest import TestCase
import numpy as np

from minf_part1_functions import find_heading, find_perpendicular_heading, is_right_of

class Test_Find_Heading(TestCase):
    def test_0_degrees(self):
        v = [42,0]
        expected = [1,0]
        actual = find_heading(v)

        np.testing.assert_array_equal(expected, actual)

    def test_random_values(self):
        for i in range(100):
            v = np.random.rand(2)
            expected = v / np.linalg.norm(v)
            actual = find_heading(v)

            np.testing.assert_array_equal(expected, actual)

class Test_Find_Perpendicular_Heading(TestCase):
    def test_0_degrees(self):
        v = [0,1]
        expected = [1,0]
        actual = find_perpendicular_heading(v)

        np.testing.assert_array_equal(expected, actual)
    
    def test_90_degrees(self):
        v = [1,0]
        expected = [0,-1]
        actual = find_perpendicular_heading(v)

        np.testing.assert_array_equal(expected, actual)

    def test_180_degrees(self):
        v = [0,-1]
        expected = [-1,0]
        actual = find_perpendicular_heading(v)

        np.testing.assert_array_equal(expected, actual)

    def test_270_degrees(self):
        v = [-1,0]
        expected = [0,1]
        actual = find_perpendicular_heading(v)

        np.testing.assert_array_equal(expected, actual)


    def test_no_direction(self):
        v = [0,0]
        expected = [0,0]
        actual = find_perpendicular_heading(v)

        np.testing.assert_array_equal(expected, actual)

class TestIsRightOf(TestCase):
    def test_0_degree_true(self):
        v = [0,1]
        p1 = [1,0]
        p2 = [-1,0]
        self.assertTrue(is_right_of(v, p1, p2))

    def test_0_degree_false(self):
        v = [0,1]
        p1 = [-1,0]
        p2 = [1,0]
        self.assertFalse(is_right_of(v, p1, p2))

    def test_0_degree_inline(self):
        v = [0,1]
        p1 = [0,0]
        p2 = [0,-1]

        self.assertIsNone(is_right_of(v, p1, p2))

    def test_90_degree_true(self):
        v = [1,0]
        p1 = [0,-1]
        p2 = [0,1]
        self.assertTrue(is_right_of(v, p1, p2))

    def test_90_degree_false(self):
        v = [1,0]
        p1 = [0,1]
        p2 = [0,-1]

        self.assertFalse(is_right_of(v, p1, p2))

    def test_180_degree_true(self):
        v = [0,-1]
        p1 = [-1,0]
        p2 = [1,0]
        self.assertTrue(is_right_of(v, p1, p2))

    def test_180_degree_false(self):
        v = [0,-1]
        p1 = [1,0]
        p2 = [-1,0]
        self.assertFalse(is_right_of(v, p1, p2))

    def test_270_degree_true(self):
        v = [-1,0]
        p1 = [0,1]
        p2 = [0,-1]
        self.assertTrue(is_right_of(v, p1, p2))

    def test_270_degree_false(self):
        v = [-1,0]
        p1 = [0,-1]
        p2 = [0,1]

        self.assertFalse(is_right_of(v, p1, p2))