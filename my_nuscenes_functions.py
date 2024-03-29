import numpy as np
from constants import *
from scipy.spatial.transform import Rotation


def get_ego_velocity(nusc, sample_token):
    """
    Finds the velocity of the ego in a given sample

    Adapted from NuScenes SDK method NuScenes.box_velocity() (https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py)

        Parameters:
            nusc: NuScenes object
            sample_token (str): sample_token to calculate velocity for
        Returns:
            velocity: velocity of ego

    """
    velocity = np.zeros(3)

    sample_data_token = nusc.get('sample', sample_token)['data']['CAM_FRONT']
    sample_data = nusc.get('sample_data', sample_data_token)
    sample_data_token_prev = sample_data['prev']
    sample_data_token_next = sample_data['next']

    # cannot calculate velocity if there are no adjacent samples
    if not (sample_data_token_prev or sample_data_token_next):
        velocity[:] = np.nan
        return velocity

    # use this sample if only one adjacent sample
    if not sample_data_token_prev:
        sample_data_token_prev = sample_data_token
    if not sample_data_token_next:
        sample_data_token_next = sample_data_token

    sample_data_prev = nusc.get('sample_data', sample_data_token_prev)
    sample_data_next = nusc.get('sample_data', sample_data_token_next)

    ego_pose_prev = nusc.get('ego_pose', sample_data_prev['ego_pose_token'])
    ego_pose_next = nusc.get('ego_pose', sample_data_next['ego_pose_token'])

    delta_translation = np.array(ego_pose_next['translation']) - np.array(ego_pose_prev['translation'])
    delta_time = (ego_pose_next['timestamp'] - ego_pose_prev['timestamp']) * 1e-6
    velocity = delta_translation / delta_time

    return velocity


def get_delta_translation(nusc, annotation, heading_angle):
    """
    Takes an annotation dictionary and finds the translation between it and the ego.

    See find_translation() for the method of calculating translation.

        Parameters:
            heading_angle: Heading of ego anti-clockwise from north
            nusc: NuScenes object
            annotation: dictionary annotation
        Returns:
            translation: numpy array in form [x,y]
    """
    sample = nusc.get('sample', annotation['sample_token'])
    sample_data_token = sample['data']['RADAR_FRONT']
    sample_data = nusc.get('sample_data', sample_data_token)
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    ego_trans = np.array(ego_pose['translation'])[0:2]

    w = renault_zoe_dims['width']
    l = renault_zoe_dims['length']

    r = ego_pose['rotation']
    r = [r[1], r[2], r[3], r[0]]  # convert to scalar last as required by SciPy
    r = Rotation.from_quat(r)
    angle = r.as_euler('xyz')[2]  # lowercase xyz for extrinsic, take the rotation around z axis

    # find offset to front of car
    calibrated_sensor = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    offset = calibrated_sensor['translation'][0]

    front_right = rotation(angle, np.array([offset, -w / 2]))
    front_left = rotation(angle, np.array([offset, w / 2]))
    back_left = rotation(angle, np.array([offset - l, w / 2]))
    back_right = rotation(angle, np.array([offset - l, -w / 2]))

    ego_bb = [ego_trans + front_right,  # front-right
              ego_trans + front_left,  # front-left
              ego_trans + back_left,  # back-left
              ego_trans + back_right]  # back-right

    ego_bb = np.array(ego_bb)

    ann_bb = nusc.get_box(annotation['token']).bottom_corners()[0:2]
    ann_bb = np.array(list(zip(ann_bb[0], ann_bb[1])))

    return find_translation(ego_bb, ann_bb, heading_angle)


def find_translation(ego_bb, ann_bb, heading_angle):
    """
    Finds the translation between the ego and annotation.

    Note: Translation is the minimum longitudinal and lateral distances between
    the ego and the other object. These distances are calculated individually.

    Args:
        ego_bb: bounding box of ego
        ann_bb: bounding box of annotation
        heading_angle: heading of the ego in radians

    Returns: translation between objects

    """

    ego_bb = np.array(ego_bb)
    ann_bb = np.array(ann_bb)

    # rotate all points to be aligned with ego
    for i in range(4):
        ego_bb[i] = rotation(-heading_angle, ego_bb[i])
        ann_bb[i] = rotation(-heading_angle, ann_bb[i])

    # find the bounds of the ego
    ego_left = min([point[0] for point in ego_bb])
    ego_right = max([point[0] for point in ego_bb])
    ego_front = max([point[1] for point in ego_bb])
    ego_back = min([point[1] for point in ego_bb])

    # find the bounds of the annotation
    ann_left = min([point[0] for point in ann_bb])
    ann_right = max([point[0] for point in ann_bb])
    ann_front = max([point[1] for point in ann_bb])
    ann_back = min([point[1] for point in ann_bb])

    x = find_dist_between_ranges([ego_left, ego_right], [ann_left, ann_right])
    y = find_dist_between_ranges([ego_back, ego_front], [ann_back, ann_front])

    translation = [x, y]
    translation = rotation(heading_angle, translation)

    return translation


def find_dist_between_ranges(range_1, range_2):
    """
    Finds the distance between two ranges.

    If ranges overlap, then the distance is zero,
    otherwise is the distance between the edges.

    Args:
        range_1: first 1-D list of length 2
        range_2: second 1-D list of length 2

    Returns: distance between ranges

    """

    # Sort the values in each range by size [small, big]
    # Then find the distance between the ranges
    range_1.sort()
    range_2.sort()

    if range_1[1] < range_2[0]:
        return range_2[0] - range_1[1]
    elif range_1[0] > range_2[1]:
        return range_2[1] - range_1[0]
    else:
        return 0


def rotation(a, p):
    """
    Rotates the 2-dimensional input point anti-clockwise by the input value
    Args:
        a: rotation point
        p: input vector

    Returns: rotated point

    """
    p = np.array(p)[0:2]
    matrix = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return matrix.dot(p)


def get_ego_heading(nusc, sample_token):
    """
    Finds the heading of the car at given sample

    Args:
        nusc: NuScenes object
        sample_token: token of sample

    Returns: heading of car

    """

    sample = nusc.get('sample', sample_token)
    sample_data = nusc.get('sample_data', sample['data']['RADAR_FRONT'])
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    r = ego_pose['rotation']
    r = [r[1], r[2], r[3], r[0]]  # convert to scalar last
    r = Rotation.from_quat(r)
    r = r.as_euler('xyz')[2] - (np.pi / 2)  # convert to from y-axis
    r = r % (2 * np.pi)
    return r
