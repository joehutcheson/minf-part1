import numpy as np

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
            translation: numpy array in form [x,y,z]
    '''
    sample = nusc.get('sample', annotation['sample_token'])
    sample_data_token = sample['data']['CAM_FRONT']
    sample_data = nusc.get('sample_data', sample_data_token)
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])

    ego_translation = np.array(ego_pose['translation'])
    annotation_translation = np.array(annotation['translation'])
    annotation_translation[2] = 0  # set z coordinate to zero

    return annotation_translation - ego_translation