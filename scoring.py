from numpy.linalg import norm
from my_nuscenes_functions import *
from constants import *


def generate_scores_for_scene(nusc, scene_token, aggressive=True):
    """
    Identifies dangerous scenarios in a scene
    Args:
        nusc (NuScenes): a nuScenes object
        scene_token (str): the token of the scene to analyse
        aggressive: If true, then aggressive RSS parameters are used, otherwise conservative parameters are used

    Returns:
        scores: list of score dictionaries

    """

    scores = []

    scene = nusc.get('scene', scene_token)

    sample = nusc.get('sample', scene['first_sample_token'])

    # set stores instances that have been processed
    instances = set()

    done = False
    # loop through all samples in scenes
    while not done:
        # loop through all annotations in sample
        for ann in sample['anns']:
            instance_token = nusc.get('sample_annotation', ann)['instance_token']
            instance = nusc.get('instance', instance_token)
            category = nusc.get('category', instance['category_token'])['name']
            if instance_token not in instances and 'vehicle' in category:
                s = generate_scores_for_instance(nusc, instance_token, aggressive=aggressive)
                if s:
                    s = min(s, key=lambda score_dict: score_dict['score'])  # get minimum score
                    s['instance'] = instance_token
                    scores.append(s)
                    instances.add(instance_token)

        # check for next sample
        if sample['next']:
            sample = nusc.get('sample', sample['next'])
        else:
            done = True

    return scores


def generate_scores_for_instance(nusc, instance_token, aggressive=True):
    """
    Returns a list of scores for an interaction with an instance
        Parameters:
            nusc (NuScenes): NuScenes object
            instance_token (str): Token of instance
            aggressive: If true, then aggressive RSS parameters are used, otherwise conservative parameters are used

        Returns:
            scores: list of dict containing
                annotation: annotation token
                reason: reason for score. If perfect score then None
                score: from 0 to 1 to describe safety across scene
    """

    if aggressive:
        params = rss_aggressive
    else:
        params = rss_conservative

    first_annotation_token = nusc.get('instance', instance_token)['first_annotation_token']
    annotation = nusc.get('sample_annotation', first_annotation_token)
    scores = []
    next_annotation = True
    while next_annotation:
        score = generate_score_for_annotation(annotation, nusc, params, scores)

        if score:
            scores.append(score)

        # continue to the next annotation
        if not annotation['next']:
            next_annotation = False
        else:
            annotation = nusc.get('sample_annotation', annotation['next'])
    return scores


def generate_score_for_annotation(annotation, nusc, params, scores):
    # check vehicle is not parked
    if any([nusc.get('attribute', t)['name'] == 'vehicle.parked'
            or nusc.get('attribute', t)['name'] == 'cycle.without_rider'
            for t in annotation['attribute_tokens']]):
        return None

    # find velocities of ego and annotated vehicle
    v_ego = get_ego_velocity(nusc, annotation['sample_token'])
    v_ann = nusc.box_velocity(annotation['token'])

    # Check all velocities are valid
    if np.isnan(v_ego).any() or np.isnan(v_ann).any():
        return None

    # find the longitudinal and lateral velocities w.r.t the heading of the ego
    heading_angle = get_ego_heading(nusc, annotation['sample_token'])

    v_ego_aligned = rotation(-heading_angle, v_ego)
    v_ann_aligned = rotation(-heading_angle, v_ann)

    # Not yet handling reversing
    if v_ego_aligned[1] < 0:
        return None

    translation = get_delta_translation(nusc, annotation, heading_angle)

    # check the relative positions of the vehicles
    ego_is_behind = is_right_of(-heading_angle + np.pi / 2, np.zeros(2), translation)
    ego_is_right = is_right_of(-heading_angle, np.zeros(2), translation)

    same_direction = None

    if ego_is_behind:
        # find the longitudinal distance between the vehicles w.r.t the
        # heading of the ego
        d_long = np.abs(rotation(-heading_angle, translation)[1])
        # find the minimum longitudinal distance between the cars
        if v_ann_aligned[1] >= 0:
            # cars travelling in same direction
            same_direction = True
            d_long_min = find_min_long_distance(norm(v_ego_aligned[1]),
                                                norm(v_ann_aligned[1]),
                                                params)
        else:
            # cars travelling in opposite directions
            same_direction = False
            d_long_min = find_min_long_distance_opposite_direction(norm(v_ego_aligned[1]),
                                                                   -norm(v_ann_aligned[1]),
                                                                   params)
        long_score = generate_individual_score(d_long_min, d_long, gradient=0.4)
    else:
        # ego doesn't hold responsibility for vehicle behind
        long_score = 1
        d_long = None
        d_long_min = None

    # find the lateral distance between the vehicles w.r.t the heading of the ego
    d_lat = np.abs(rotation(-heading_angle, translation)[0])

    # Assign velocities to input variables for RSS rule 2.
    # c1 is on the left, c2 is on the right, with velocities v1 and v2 respectively
    if ego_is_right:
        v1 = v_ann_aligned[0]
        v2 = v_ego_aligned[0]
    else:
        v1 = v_ego_aligned[0]
        v2 = v_ann_aligned[0]

    # find the minimum lateral distances between the cars
    d_lat_min = find_min_lat_distance(v1,
                                      v2,
                                      params)
    lat_score = generate_individual_score(d_lat_min, d_lat, gradient=1)

    # if either the lateral distance or the longitudinal distance is okay,
    # then we consider the situation safe, hence max of each score is used
    max_score = max([long_score, lat_score])

    # note the reason for the score
    if max_score == 1:
        reason = None
    elif max_score == 0:
        reason = 'Too close'
    elif max_score == long_score:
        reason = 'Longitudinally too close'
    elif max_score == lat_score:
        reason = 'Laterally too close'
    else:
        reason = 'Unknown'

    # Copy reason from last iteration if possible when current reason is unhelpful
    if reason == 'Too close' and len(scores) > 0:
        if scores[-1]['reason'] is not None:
            reason = scores[-1]['reason']

    return {
        'annotation': annotation['token'],
        'reason': reason,
        'score': max_score,
        'ego_long_velocity': v_ego_aligned[1],
        'ego_lat_velocity': v_ego_aligned[0],
        'ann_long_velocity': v_ann_aligned[1],
        'ann_lat_velocity': v_ann_aligned[0],
        'long_distance': rotation(-heading_angle, translation)[1],
        'lat_distance': rotation(-heading_angle, translation)[0],
        'min_long_distance': d_long_min,
        'min_lat_distance': d_lat_min,
        'same_direction': same_direction
    }


def find_min_long_distance(v_r, v_f, params):
    """
    Returns the minimum required longitudinal distance per Rule 1 of RSS

        Parameters:
            v_r: Velocity of rear vehicle
            v_f: Velocity of front vehicle
            params: Parameter dictionary to use

        Returns:
            d_min: Minimum required distance
    """

    assert v_r >= 0
    assert v_f >= 0

    # find constant parameters
    a_max_accel = params['a_long_max_accel']
    a_min_brake = params['a_long_min_brake']
    a_max_brake = params['a_long_max_brake']
    p = params['p']

    d_min = (
            v_r * p
            + 0.5 * a_max_accel * (p ** 2)
            + ((v_r + p * a_max_accel) ** 2) / (2 * a_min_brake)
            - (v_f ** 2) / (2 * a_max_brake)
    )

    return max([0, d_min])


def find_min_long_distance_opposite_direction(v_1, v_2, params):
    """
    Finds the minimum longitudinal distance with must be maintained between two cars
    travelling towards each other.

    Args:
        v_1: Velocity of car with positive velocity
        v_2: Velocity of car with negative velocity
        params: Parameter dictionary to use

    Returns: Minimum distance required

    """

    assert v_1 >= 0
    assert v_2 < 0

    # find constant parameters
    a_min_brake = params['a_long_min_brake']
    a_min_brake_corr = params['a_long_min_brake_correct']
    a_max_accel = params['a_long_max_accel']
    p = params['p']

    v_1_p = v_1 + p * a_max_accel
    v_2_p = abs(v_2) + p * a_max_accel

    d_min = (
            (v_1 + v_1_p) * p / 2
            + (v_1_p ** 2) / (2 * a_min_brake_corr)
            + (abs(v_2) + v_2_p) * p / 2
            + (v_2_p ** 2) / (2 * a_min_brake)
    )

    return d_min


def find_min_lat_distance(v_1, v_2, params):
    """
    Returns the minimum required lateral distance per Rule 2 of RSS. All parameters having right heading

        Parameters:
            v_1: lateral velocity of left car (positive to the right)
            v_2: lateral velocity of right car (positive to the left)
            params: Parameter dictionary to use

        Returns:
            d_min: minimum safe lateral distance

    """

    # find constant parameters
    mu = params['mu']
    a_lat_min_brake = params['a_lat_min_brake']
    a_lat_max_accel = params['a_lat_max_accel']
    p = params['p']

    # if the cars are moving apart then situation is not dangerous
    if v_1 < 0 or v_2 > 0:
        return mu

    v_1_p = v_1 + p * a_lat_max_accel
    v_2_p = v_2 - p * a_lat_max_accel

    d_min = (
            ((v_1 + v_1_p) * p) / 2
            + (v_1_p ** 2) / (2 * a_lat_min_brake)
            - (
                    ((v_2 + v_2_p) * p) / 2
                    - (v_2_p ** 2) / (2 * a_lat_min_brake)
            )
    )

    return max([mu, mu + d_min])


def generate_individual_score(minimum, actual, gradient):
    """
    Generates a score between 0 and 1 given minimum and actual values

        Parameters:
            minimum: minimum allowable value
            actual: actual value
            gradient: measure of how close the values can be. Higher value gives higher score when close

        Returns:
            score: the generated score
    """

    margin = actual - minimum

    if margin <= 0:
        return 0

    score = gradient * margin

    return min(1, score)  # score cannot be greater than 1


def is_right_of(theta, p1, p2):
    """
    Finds out if a point is to the right of another w.r.t the direction given by v

        Parameters:
            theta: heading angle
            p1: first point
            p2: second point

        Returns:
            result: True if p1 is right of p2, None if directly inline, False otherwise
    """

    # ensure only 2-dimensional
    p1 = p1[0:2]
    p2 = p2[0:2]

    # rotate the positions
    p1 = rotation(theta, p1)
    p2 = rotation(theta, p2)

    # compare x values to determine left and right
    if p1[0] > p2[0]:
        return True
    if p1[0] < p2[0]:
        return False
    return None
