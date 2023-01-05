import numpy as np
from numpy.linalg import norm
from my_nuscenes_functions import *
from constants import *


def generate_scores_for_instance(nusc, instance_token, aggressive=False):
    '''
    Returns a list of scores for an interaction with an instance
        Parameters:
            nusc (NuScenes): NuScenes object
            instance_token (str): Token of instance

        Returns:
            scores: list of dict containing
                annotation: annotation token
                reason: reason for score. If perfect score then None
                score: from 0 to 1 to describe saftey across scene
    '''
    
    if aggressive:
        params=rss_aggressive
    else:
        params=rss_conservative
    
    first_annotation_token = nusc.get('instance', instance_token)['first_annotation_token']
    annotation = nusc.get('sample_annotation', first_annotation_token)
    scores = []
    next_annotation = True
    while next_annotation:
        # TODO: Make sure the longitudnal velocity is in relation to the road not the ego heading
        
        # find velocities of ego and annotated vehicle
        v_ann = nusc.box_velocity(annotation['token'])
        v_ego = get_ego_velocity(nusc, annotation['sample_token'])

        # find the heading of the ego
        ego_heading = find_heading(v_ego)
        if ego_heading is None:
            max_score = 1 # ego is stopped
        else:
            # find the perpendicular heading of the ego
            perp_ego_heading = find_perpendicular_heading(ego_heading)

            # find the longitudnal and lateral velocities w.r.t the heading of the ego
            v_ego_long = v_ego[0:2] * ego_heading
            v_ego_lat = v_ego[0:2] * perp_ego_heading
            v_ann_long = v_ann[0:2] * ego_heading
            v_ann_lat = v_ann[0:2] * perp_ego_heading
            
            translation = get_delta_translation(nusc, annotation)

            # check the relative postitions of the vehicles
            ego_is_behind = isRightOf(perp_ego_heading, np.zeros(2), translation)
            ego_is_right = isRightOf(ego_heading, np.zeros(2), translation)

            if ego_is_behind:
                # find the longitudnal distance between the vehicles w.r.t the
                # heaidng of the ego
                d_long = np.abs(norm(translation * ego_heading))
                # find the minimum longitudnal distance between the cars
                if angle_between(v_ego_long, v_ann_long) < np.pi / 2:
                    # cars in same direction
                    d_long_min = find_min_long_distance(norm(v_ego_long), 
                                                norm(v_ann_long),
                                                params)
                else:
                    # cars in oposite direcitons
                    d_long_min = find_min_long_distance_oposite_direction(norm(v_ego_long),
                                                                          norm(v_ann_long),
                                                                          params)
                long_score = generate_indivual_score(d_long_min, d_long)
            else:
                # ego doesn't hold responsibility for vehicle behind
                long_score = 1 
            
            # find the lateral distance between the vehicles w.r.t the
            # heaidng of the ego
            d_lat = np.abs(norm(translation * perp_ego_heading))

            # find angle from the x-axis of the perpendicular heading
            d_lat_theta = np.arctan2(perp_ego_heading[0], perp_ego_heading[1])

            # rotate the lateral velocities to the x-axis
            v_ego_lat_single_value = rotation(d_lat_theta, v_ego_lat)[0]
            v_ann_lat_single_value = rotation(d_lat_theta, v_ann_lat)[0]

            # check and assign relative positions of vehicles
            if ego_is_right:
                v1 = v_ann_lat_single_value
                v2 = v_ego_lat_single_value
            else:
                v1 = v_ego_lat_single_value
                v2 = v_ann_lat_single_value
            # find the minimum lateral distances between the cars
            d_lat_min = find_min_lat_distance(v1,
                                            v2,
                                            params) 
            lat_score = generate_indivual_score(d_lat_min, d_lat, strictness=4)
            
            # if either the lateral distance or the longitudal distance is okay,
            # then we consider the situation safe, hence max of each score is used
            max_score = max([long_score, lat_score])

        # note the reason for the score
        if max_score == 1:
            reason = None
        elif max_score == long_score:
            reason = 'Longitudnally too close'
        elif max_score == lat_score:
            reason = 'Laterally too close'
        scores.append({
            'annotation':annotation['token'], 
            'reason': reason,
            'score': max_score
            })

        # continue to the next annotation
        if not annotation['next']:
            next_annotation = False
        else:
            annotation = nusc.get('sample_annotation', annotation['next'])
    return scores


# Finds the minimum required longitudnal distance between cars by RSS Rule 1
# Inputs: See RSS
# Output: Minimum required distance
def find_min_long_distance(v_r, v_f, 
                           params=None,
                           a_max_accel=2.44,
                           a_min_brake=3.2,
                           a_max_brake=8.2,
                           p=0.5):
    '''
    Returns the minimum required longitudnal distance per Rule 1 of RSS

        Parameters:
            v_r: Velocity of rear vehicle
            v_f: Velocity of front vehicle
            a_max_accel: Maximum expected acceleration of rear car
            a_min_brake: Minimum acceleration due to braking of rear car
            a_max_brake: Maximum acceleration due to braking of front car
        
        Returns:
            d_min: Minimum requred distance
    '''
    
    if params != None:
        a_max_accel = params['a_long_max_accel']
        a_min_brake = params['a_long_min_brake']
        a_max_brake = params['a_long_max_brake']
        p = params['p']
    
    d_min = (
            v_r*p
            + 0.5*a_max_accel*(p**2)
            + ((v_r + p*a_max_accel)**2)/(2*a_min_brake)
            - (v_f**2)/(2*a_max_brake)
    )
    
    return max([0,d_min])

def find_min_long_distance_oposite_direction(v_1, v_2, params):
    a_min_brake = params['a_long_min_brake']
    a_min_brake_corr = params['a_long_min_brake_correct']
    a_max_accel = params['a_long_max_accel']
    p = params['p']

    v_1_p = v_1 + p * a_max_accel
    v_2_p = abs(v_2) + p * a_max_accel

    d_min = (
        (v_1 + v_1_p) * p / 2
        + v_1_p ** 2 / 2 * a_min_brake_corr
        + (abs(v_2) + v_2_p) * p / 2
        + v_2_p ** 2 / 2 * a_min_brake
    )

    return d_min

def find_min_lat_distance(v_1, v_2,
                          params=None,
                          mu=0.1, 
                          a_lat_min_brake=1.0, 
                          a_lat_max_accel=1.0,
                          p=0.5):
    '''
    Returns the minimum required lateral distance per Rule 2 of RSS. All parameters having right heading

        Parameters:
            v_1: lateral velocity of left car (positive to the right)
            v_2: lateral velocity of right car (positive to the left)
            mu: Minimum required distance between the cars
            a_lat_min_brake: the minimum acceleration each car will apply apart from each other
            a_lat_max_accel: the maximum acceleration each car will initially apply towards each other
            p: response time

        Returns:
            d_min: minimum safe lateral distance

    '''
    
    if params != None:
        mu = params['mu']
        a_lat_min_brake = params['a_lat_min_brake']
        a_lat_max_accel = params['a_lat_max_accel']
        p = params['p']

    v_1_p = v_1 + p * a_lat_max_accel
    v_2_p = v_2 - p * a_lat_max_accel
    

    d_min = (
              ((v_1 + v_1_p) * p) / 2
              + (v_1_p**2) / (2 * a_lat_min_brake)
              - (
                ((v_2 + v_2_p) * p) / 2
                - (v_2_p**2)/(2 * a_lat_min_brake)
              )
            )

    return max([mu, mu + d_min])



def generate_indivual_score(minimum, actual, strictness=0.1):
    '''
    Generates a score between 0 and 1 given minimum and actual values

        Parameters:
            minimum: minimum allowable value
            actual: actual value
            strictness: measure of how close the values can be. Higher value gives higher score when close

        Returns:
            score: the generated score
    '''

    margin = actual - minimum

    if margin <= 0:
        return 0

    score = strictness * margin

    return min(1, score)  # score cannot be greater than 1

def find_heading(v):
    '''
    Find the unit vector as a heading of the input vector

        Parameters:
            v: input vector

        Returns:
            heading
    '''
    v = np.array(v)
    v = v[0:2] # ensure 2-dimensional
    magnitude = norm(v)

    if magnitude == 0:
        return None

    return v / magnitude

def find_perpendicular_heading(heading):
    '''
    Takes a heading vector and finds the vector rotated 90 degrees around z axis

        Parameters:
            heading: heading vector
        
        Returns:
            perpendicular_heading
    '''
    return np.array([heading[1], -heading[0]])

def rotation(a, p):
    a = np.array(a)
    p = np.array(p)
    R = np.array([[np.cos(a),-np.sin(a)],[np.sin(a), np.cos(a)]])
    return R.dot(p)


def isRightOf(v, p1, p2):
    '''
    Finds out if a point is to the right of another w.r.t the direction given by v

        Parameters:
            v: heading_vector
            p1: first point
            p2: second point

        Returns:
            result: True if p1 is right of p2, None if directly inline, False otherwise
    '''


    # ensure only 2-dimensional
    v = v[0:2]
    p1 = p1[0:2]
    p2 = p2[0:2]

    # calculate anti clockwise angle of heading from y axis
    # TODO is this the y-axis?? or is it the x-axis? It might not matter...
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