from unittest import TestCase
import numpy as np

from my_nuscenes_functions import find_translation, rotation

class Test_Find_Translation(TestCase):
    def test_1(self):
        # Two boxes placed beside each other. 
        ego_bb = np.array([[0,0],[1,0],[1,1],[0,1]])
        ann_bb = np.array([[2,0],[3,0],[3,1],[2,1]])
        expected = [1,0]

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
        ego_bb = np.array([[0,0],[1,0],[1,1],[0,1]])
        ann_bb = np.array([[0,2],[1,2],[1,3],[0,3]])
        expected = [0,1]

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
        ego_bb = np.array([[3,3],[4,3],[4,4],[3,4]])
        ann_bb = np.array([[1,1],[2,1],[2,2],[1,2]])
        expected = [-1,-1]

        for _ in range(4):
            actual = find_translation(ego_bb, ann_bb)
            np.testing.assert_allclose(expected, actual)
            ego_bb = np.roll(ego_bb, 1, axis=0)

        for _ in range(4):
            actual = find_translation(ego_bb, ann_bb)
            np.testing.assert_allclose(expected, actual)
            ann_bb = np.roll(ann_bb, 1, axis=0)

class Test_Rotation(TestCase):
    def test_90_degree_1(self):
        angle = np.pi/2
        vector = [0,1]
        expected = np.array([-1, 0])
        actual = rotation(angle, vector)
        np.testing.assert_allclose(expected, actual, atol=0.00001)