import numpy as np
import math
from constants import *
from scipy.spatial.transform import Rotation

def get_ego_velocity(nusc, sample_token):
    '''
    Finds the velocity of the ego in a given sample

    Adapted from NuScenes SDK method NuScenes.box_velocity() (https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py)
    
        Parameters:
            nusc: NuScenes object
            sample_token (str): sample_token to calculate velocity for
        Returns:
            velocity: velocity of ego
    
    '''
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
    assert delta_time != 0 # to avoid divide by zero error
    velocity = delta_translation / delta_time

    return velocity


def get_delta_translation(nusc, annotation):
    '''
    Takes an annotation dictionary and finds the translation between it and the ego

        Parameters:
            nusc: NuScenes object
            annotation: dictionary annotation
        Returns:
            translation: numpy array in form [x,y]
    '''
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

    front_right = rotation(angle, np.array([offset,-w/2]))
    front_left = rotation(angle, np.array([offset, w/2]))
    back_left = rotation(angle, np.array([offset-l,w/2]))
    back_right = rotation(angle, np.array([offset-l,-w/2]))
    
    ego_bb = [ego_trans + front_right,  # front-right
              ego_trans + front_left,  # front-left
              ego_trans + back_left,  # back-left
              ego_trans + back_right]  # back-right

    ego_bb = np.array(ego_bb)
              
    
    ann_bb = nusc.get_box(annotation['token']).bottom_corners()[0:2]
    ann_bb = np.array(list(zip(ann_bb[0], ann_bb[1])))
    
    return find_translation(ego_bb, ann_bb)
    

def find_translation(ego_bb, ann_bb):

    ego_bb = np.array(ego_bb)
    ann_bb = np.array(ann_bb)

    # https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    def length_between_line_and_point(p, l):
        if math.dist(l[0], l[1]) == 0:
            return math.dist(l[0], p)
        l_squared = np.linalg.norm(l[0]-l[1])**2
        t = max(0, min(1, np.dot(p - l[0], (l[1] - l[0])/l_squared)))
        projection = l[0] + (t * (l[1] - l[0]))
        return math.dist(p, projection), projection
        
        
    # find min distance between corners and edges of the boxes
    min_dist, projection = length_between_line_and_point(ego_bb[0], ann_bb[0:2])
    min_trans = projection - ego_bb[0]
    for i in range(4):
        p1 = ego_bb[i]
        p2 = ego_bb[(i+1)%4]
        l = np.array([p1,p2])
        for j in range(4):
            p = ann_bb[j]
            dist, projection = length_between_line_and_point(p, l)
            if dist < min_dist:
                min_dist = dist
                min_trans = p - projection
                
    for i in range(4):
        p1 = ann_bb[i]
        p2 = ann_bb[(i+1)%4]
        l = np.array([p1,p2])
        for j in range(4):
            p = ego_bb[j]
            dist, projection = length_between_line_and_point(p, l)
            if dist < min_dist:
                min_dist = dist
                min_trans = projection - p
            
    return min_trans

def rotation(a, p):
    """
    Rotates the 2-dimensional input vector anti-clockwise by the input value
    Args:
        a: rotation angle
        p: input vector

    Returns:

    """
    p = np.array(p)[0:2]
    matrix = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return matrix.dot(p)

def get_car_heading(nusc, sample_token):
    sample = nusc.get('sample', sample_token)
    sample_data = nusc.get('sample_data', sample['data']['RADAR_FRONT'])
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    r = ego_pose['rotation']
    r = [r[1], r[2], r[3], r[0]] # convert to scalar last
    r = Rotation.from_quat(r)
    r = r.as_euler('xyz')[2] - (np.pi/2) # convert to from y-axis
    r = r % (2*np.pi)
    return r