from numpy.linalg import norm
from my_nuscenes_functions import *
from constants import *


def generate_scores_for_scene(nusc, scene_token):
    scores = []

    scene = nusc.get('scene', scene_token)

    sample = nusc.get('sample', scene['first_sample_token'])

    def get_score(score_dict):
        return score_dict['score']

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
                s = generate_scores_for_instance(nusc, instance_token, aggressive=True)
                if s:
                    s = min(s, key=get_score)
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
        # TODO: Make sure the longitudinal velocity is in relation to the road not the ego heading

        # check vehicle is not parked
        if not any([nusc.get('attribute', t)['name'] == 'vehicle.parked'
                    or nusc.get('attribute', t)['name'] == 'cycle.without_rider'
                    for t in annotation['attribute_tokens']]):

            # find velocities of ego and annotated vehicle
            v_ego = get_ego_velocity(nusc, annotation['sample_token'])
            v_ann = nusc.box_velocity(annotation['token'])

            # find the longitudinal and lateral velocities w.r.t the heading of the ego
            heading_angle = get_car_heading(nusc, annotation['sample_token'])
            heading_vector = angle_to_vector(heading_angle)

            v_ego_aligned = rotation(-heading_angle, v_ego)
            v_ann_aligned = rotation(-heading_angle, v_ann)

            v_ego_long = rotation(heading_angle, [0,v_ego_aligned[1]])
            v_ego_lat = rotation(heading_angle, [v_ego_aligned[0],0])
            v_ann_long = rotation(heading_angle, [0,v_ann_aligned[1]])
            v_ann_lat = rotation(heading_angle, [v_ann_aligned[0],0])

            translation = get_delta_translation(nusc, annotation)

            # check the relative positions of the vehicles
            ego_is_behind = is_right_of(find_perpendicular_heading(heading_vector), np.zeros(2), translation)
            ego_is_right = is_right_of(heading_vector, np.zeros(2), translation)

            if ego_is_behind:
                # find the longitudinal distance between the vehicles w.r.t the
                # heading of the ego
                d_long = np.abs(rotation(-heading_angle, translation)[1])
                # find the minimum longitudinal distance between the cars
                if v_ann_aligned[1] >= 0:
                    # cars travelling in same direction
                    d_long_min = find_min_long_distance(norm(v_ego_long),
                                                        norm(v_ann_long),
                                                        params)
                else:
                    # cars travelling in opposite directions
                    d_long_min = find_min_long_distance_opposite_direction(norm(v_ego_long),
                                                                           -norm(v_ann_long),
                                                                           params)
                long_score = generate_individual_score(d_long_min, d_long)
            else:
                # ego doesn't hold responsibility for vehicle behind
                long_score = 1

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
            lat_score = generate_individual_score(d_lat_min, d_lat, strictness=1)

            # if either the lateral distance or the longitudinal distance is okay,
            # then we consider the situation safe, hence max of each score is used
            max_score = max([long_score, lat_score])

            # note the reason for the score
            if max_score == 1:
                reason = None
            elif max_score == long_score:
                reason = 'Longitudinally too close'
            elif max_score == lat_score:
                reason = 'Laterally too close'
            else:
                reason = 'Unknown'
            scores.append({
                'annotation': annotation['token'],
                'reason': reason,
                'score': max_score
            })

        # continue to the next annotation
        if not annotation['next']:
            next_annotation = False
        else:
            annotation = nusc.get('sample_annotation', annotation['next'])
    return scores


# Finds the minimum required longitudinal distance between cars by RSS Rule 1
# Inputs: See RSS paper
# Output: Minimum required distance
def find_min_long_distance(v_r, v_f,params):
    """
    Returns the minimum required longitudinal distance per Rule 1 of RSS

        Parameters:
            v_r: Velocity of rear vehicle
            v_f: Velocity of front vehicle
            params: Parameter dictionary to use
            p: Reaction time
            a_max_accel: Maximum expected acceleration of rear car
            a_min_brake: Minimum acceleration due to braking of rear car
            a_max_brake: Maximum acceleration due to braking of front car

        Returns:
            d_min: Minimum required distance
    """

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
    assert v_1 >= 0
    assert v_2 < 0

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
            mu: Minimum required distance between the cars
            a_lat_min_brake: the minimum acceleration each car will apply apart from each other
            a_lat_max_accel: the maximum acceleration each car will initially apply towards each other
            p: response time

        Returns:
            d_min: minimum safe lateral distance

    """


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


def generate_individual_score(minimum, actual, strictness=0.2):
    """
    Generates a score between 0 and 1 given minimum and actual values

        Parameters:
            minimum: minimum allowable value
            actual: actual value
            strictness: measure of how close the values can be. Higher value gives higher score when close

        Returns:
            score: the generated score
    """

    margin = actual - minimum

    if margin <= 0:
        return 0

    score = strictness * margin

    return min(1, score)  # score cannot be greater than 1


def find_perpendicular_heading(heading):
    """
    Takes a heading vector and finds the vector rotated 90 degrees around z axis

        Parameters:
            heading: heading vector

        Returns:
            perpendicular_heading
    """
    return np.array([heading[1], -heading[0]])





def is_right_of(v, p1, p2):
    """
    Finds out if a point is to the right of another w.r.t the direction given by v

        Parameters:
            v: heading_vector
            p1: first point
            p2: second point

        Returns:
            result: True if p1 is right of p2, None if directly inline, False otherwise
    """

    # ensure only 2-dimensional
    v = v[0:2]
    p1 = p1[0:2]
    p2 = p2[0:2]

    # calculate clockwise angle of heading from y-axis
    theta = np.arctan2(v[0], v[1])

    p1 = rotation(theta, p1)
    p2 = rotation(theta, p2)

    # compare x values to determine left and right
    if p1[0] > p2[0]:
        return True
    if p1[0] < p2[0]:
        return False
    return None


def angle_between(a, b):
    a = np.array(a)
    b = np.array(b)
    if norm(a) * norm(b) == 0:
        return 0
    theta = np.arccos(
        a.dot(b) / (norm(a) * norm(b))
    )
    return theta

def angle_to_vector(a):
    x = -np.sin(a)
    y = np.cos(a)
    return np.array([x,y])
