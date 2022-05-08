import numpy as np


def clip16(x):
    '''
    Clipping for 16 bits
    '''
    if x > 32767:
        x = 32767
    elif x < -32768:
        x = -32768
    else:
        x = x
    return (x)

# Deprecated, not working well
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
        flag_inzone = False

        # Iterate through dimensions
        for idx in range(len_pt):

            # If in any dimension, point is not in the zone, set flag to false
            if point[idx] > zone_left_top[idx] and point[idx] < zone_right_bot[idx]:
                flag_inzone = True

    else:
        # If the input dimensions don't match, raise exception
        err_oper1 = 'Point dimension = {}'.format(len_pt)
        err_oper2 = 'Left top margin dimension = {}'.format(len_lt)
        err_oper3 = 'Right bottom margin dimension = {}'.format(len_rb)
        raise RuntimeError('Input dimensions doesn\'t match: {}, {}, {}.'.format(
            err_oper1, err_oper2, err_oper3))

    return flag_inzone


def get_coord(labels, hand, size_img, results):
    '''
    Get the coordinates corresponding to the labels
    '''

    # Declare a list for results
    output_list = []

    for element in labels:
        curr_coords = tuple(np.multiply(np.array((hand.landmark[element].x, hand.landmark[element].y)), size_img).astype(int))

        output_list.append(curr_coords)

    return output_list


def calc_dist(p1, p2):
    '''
    Calculate the distance between two points (same dimension)
    '''

    # get number of dimensions
    num_dim1 = len(p1)
    num_dim2 = len(p2)

    if num_dim1 == num_dim2:
        diff_list = []
        # calculate differences of coordinates
        for idx in range(num_dim1):
            diff_list.append(p1[idx] - p2[idx])

            # apply Pythagorean
        diff_arr = np.array(diff_list)
        dist = np.sqrt(np.sum(diff_arr**2))

        return dist

    else:
        raise RuntimeError('Input dimensions don\'t match.')

def set_freq(x, max_x, freq_range):
    '''
    Set the frequency for the oscillation

    x: The value that frequency change accords to.
    max_x: the largest value that x can be (do not exceed)
    f: the frequency, in Hz (in logarithmic scale of x)
    '''

    freq_range = np.array(freq_range)

    x_range = np.log10(freq_range)
    xr_width = np.max(x_range) - np.min(x_range)
    
    # get the percentage of x to max_x
    ratio = x/max_x

    # get current x
    x_curr = np.min(x_range) + xr_width * ratio

    # get current frequency
    freq_curr = 10**x_curr

    return freq_curr