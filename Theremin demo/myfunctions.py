import numpy as np


def clip16(x):
    # Clipping for 16 bits
    if x > 32767:
        x = 32767
    elif x < -32768:
        x = -32768
    else:
        x = x
    return (x)


def in_zone(point, zone_left_top, zone_right_bot):
    '''
    Determine whether a point is in a 2-D rectangular zone.

    Assume that for every dimension, zone_left_top has smallest margin and zone_right_bot has the largest margin.
    point: The point to determine.
    zone_left_top: Left top point of zone.
    zone_right_bot: Right bottom point of zone.

    Return: flag_inzone (boolean)
    '''

    # Retrieve number of dimensions
    len_pt = len(point)
    len_lt = len(zone_left_top)
    len_rb = len(zone_right_bot)

    # Check the input dimensions
    if len_pt == len_lt and len_pt == len_rb:

        # Initiate flag_inzone
        flag_inzone = True

        # Iterate through dimensions
        for idx in range(len_pt):

            # If in any dimension, point is not in the zone, set flag to false
            if point[idx] < zone_left_top[idx] or point[idx] > zone_right_bot[idx]:
                flag_inzone = False

    else:
        # If the input dimensions don't match, raise exception
        raise [RuntimeError('Input dimensions doesn\'t match.')]

    return flag_inzone


def get_coord(labels, hand, size_img):
    '''
    Get the coordinates corresponding to the labels
    '''

    # Declare a list for results
    output_list = []

    # Iterate through desired list of labels
    for element in labels:
        curr_coords = tuple(np.multiply(
            np.array((hand.landmark[element].x, hand.landmark[element].y)),
            size_img).astype(int))

        output_list.append(curr_coords)

        return output_list
