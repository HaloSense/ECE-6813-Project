import mediapipe as mp
import cv2
import pyaudio
import struct
import numpy as np
from numpy import pi

from myfunctions import *

# initialze hand recognition related
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# declare some paremeters for text printing
font = cv2.FONT_HERSHEY_TRIPLEX
font_size = 0.5
font_thickness = 1

# Output sampling rate
Fs = 8000

# Specify initial value for output flag
flag_output = False

# Create Pyaudio object
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=Fs,
    input=False,
    output=True,
    frames_per_buffer=128)

# specify output block related parameters

# for an output without artifact, block length cannot be smaller than 512
BLOCKLEN = 512

output_block = [0] * BLOCKLEN
theta = 0
f1 = 500    # temporary value for f1
gain = 12800    # temp value for gain

# some other variables to cancel artifact
gain_prev = 0
om_prev = 0

# dictionary for ranges of the zone
dict_zones = {
    'zone1': [(40, 40), (300, 420)],
    'zone2': [(340, 40), (600, 420)]
}

# dictionary to link number indices with tags
dict_hand_labels = {
    0: mp_hands.HandLandmark.WRIST,
    1: mp_hands.HandLandmark.THUMB_CMC,
    2: mp_hands.HandLandmark.THUMB_MCP,
    3: mp_hands.HandLandmark.THUMB_IP,
    4: mp_hands.HandLandmark.THUMB_TIP,
    5: mp_hands.HandLandmark.INDEX_FINGER_MCP,
    6: mp_hands.HandLandmark.INDEX_FINGER_PIP,
    7: mp_hands.HandLandmark.INDEX_FINGER_DIP,
    8: mp_hands.HandLandmark.INDEX_FINGER_TIP,
    9: mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
    10: mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
    11: mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
    12: mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    13: mp_hands.HandLandmark.RING_FINGER_MCP,
    14: mp_hands.HandLandmark.RING_FINGER_PIP,
    15: mp_hands.HandLandmark.RING_FINGER_DIP,
    16: mp_hands.HandLandmark.RING_FINGER_TIP,
    17: mp_hands.HandLandmark.PINKY_MCP,
    18: mp_hands.HandLandmark.PINKY_PIP,
    19: mp_hands.HandLandmark.PINKY_DIP,
    20: mp_hands.HandLandmark.PINKY_TIP,
}

# Capture WebCam feed
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6) as hands:
    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a selfie-view display.
        image = cv2.flip(image, 1)

        # retrieve the shape of image
        h, w, c = image.shape
        size_image = (w, h)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # Draw zones for recognizing both hands
        # Zone 1 (Left hand, controls gain)
        color_zone1 = (255, 0, 0)               # Blue for zone 1
        rect_zone1_pos1 = dict_zones['zone1'][0]
        rect_zone1_pos2 = dict_zones['zone1'][1]
        txt_pos_zone1 = (rect_zone1_pos1[0] + 10, rect_zone1_pos1[1] + 25)
        # cv2.rectangle(image, rect_zone1_pos1, rect_zone1_pos2, color_zone1, 2)
        cv2.putText(image, 'Left Hand (Gain)', txt_pos_zone1, font,
                    font_size, color_zone1, font_thickness)

        # Zone 2 (Right hand, controls frequency)
        color_zone2 = (255, 0, 255)             # Magenta for zone 2
        rect_zone2_pos1 = dict_zones['zone2'][0]
        rect_zone2_pos2 = dict_zones['zone2'][1]
        txt_pos_zone2 = (rect_zone2_pos1[0] + 10, rect_zone2_pos1[1] + 25)
        # cv2.rectangle(image, rect_zone2_pos1, rect_zone2_pos2, color_zone2, 2)
        cv2.putText(image, 'Right Hand (Freq)', txt_pos_zone2, font,
                    font_size, color_zone2, font_thickness)

        # print info about input video feed on the screen
        txt_color_imgsize = (0, 0, 255)
        txt_pos_imgsize = (10, 25)
        image = cv2.putText(image, 'Input image height: {}, width: {}. Press \"Q\" to quit.'.format(
            h, w), txt_pos_imgsize, font, font_size, txt_color_imgsize, font_thickness)

        # sound part

        # extract the landmarks that I need
        # turn the indices into labels

        # Left hand (gain): 5 landmarks on the palm (without the wrist one)
        lm_idx_left = [1, 5, 9, 13, 17]
        lm_labels_left = []
        for idx in lm_idx_left:
            lm_labels_left.append(dict_hand_labels[idx])

        # right hand (freq): 5 landmarks on the tips of fingers
        lm_idx_right = [4, 8, 12, 16, 20]
        lm_labels_right = []
        for idx in lm_idx_right:
            lm_labels_right.append(dict_hand_labels[idx])

        # extract coordinates

        # construct a 2D list to store coordinates
        hand_coords = []
        hand_coords.append([])
        hand_coords.append([])

        # get the length of multi_handedness
        # i.e. num of hands detected
        if results.multi_handedness:
            num_hands = len(results.multi_handedness)

            # output sound only when both hands are detected
            if num_hands == 2:

                # determine the indices of the hands
                # i.e. which hand has what index (0 or 1)?
                for class_idx, classification in enumerate(results.multi_handedness):
                    if classification.classification[0].index == 0:
                        left_idx = class_idx
                        right_idx = 1 - left_idx

                # append the coordinates of the hands to the list
                # hand_coords[0] is left, hand_coords[1] is right
                hand_coords[0] = get_coord(
                    lm_labels_left, results.multi_hand_landmarks[left_idx], size_image, results)
                hand_coords[1] = get_coord(
                    lm_labels_right, results.multi_hand_landmarks[right_idx], size_image, results)
                right_palm = get_coord(
                    lm_labels_left, results.multi_hand_landmarks[right_idx], size_image, results)

                # calculate the critical values:

                # left hand: average coordinate of all points
                crit_l_arr = np.array(hand_coords[0])
                crit_left = np.mean(crit_l_arr, axis=0)
                gain = (h - crit_left[1]) / h * 32767

                # right hand: average distance between points -> frequency
                crit_r_arr = np.array(hand_coords[1])
                dist_list = []
                right_palm_dist = []

                for i in range(len(crit_r_arr - 1)):
                    for j in range(i+1, len(crit_r_arr)):
                        dist_list.append(
                            calc_dist(crit_r_arr[i, :], crit_r_arr[j, :]))

                # calculate average distance
                dist_arr = np.array(dist_list)
                mean_dist = np.mean(dist_arr)
                max_dist = np.max(dist_arr)
                perc_dist = mean_dist/max_dist

                # calculate the frequency
                # essentially calculates the 
                freq_range = (200, 2000)
                f1 = set_freq(mean_dist, 300, freq_range)

                # set output flag to true
                flag_output = True

            else:
                # don't output
                flag_output = False
        else:
            flag_output = False

        # the phase omega
        om1 = 2.0 * pi * f1 / Fs

        # output when flag_output is set
        if flag_output:

            for i in range(0, BLOCKLEN):
                output_block[i] = int(clip16((gain_prev + (gain - gain_prev) * (i+1)/BLOCKLEN)* np.cos(theta)))
                theta = theta + (om_prev + (om1 - om_prev) * (i+1)/BLOCKLEN)

            # Reset theta when out of bound of pi
            if theta > pi:
                theta = theta - 2.0 * pi

            # renew buffer variables
            gain_prev = gain
            om_prev = om1

            # change output status string
            output_status = 'Active'

            # status color for active
            color_status = (0, 255, 0)

            # print gain and freq on screen

        else:
            # if not set, output zero
            output_block = [0] * BLOCKLEN
            output_status = 'Inactive'

            # status color for inactive
            color_status = (0, 0, 255)

        # output audio block
        binary_data = struct.pack(
            'h' * BLOCKLEN, *output_block)   # 'h' for 16 bits
        stream.write(binary_data)

        # show the output status on the screen
        pos_status = (20, 450)

        cv2.putText(image, 'Output Status: {}'.format(output_status),
                    pos_status, font, font_size, color_status, font_thickness)

        # Print the image
        cv2.imshow('Digital Theremin', image)

        # Wait for key 'q' to be pressed
        # When pressed, quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Exit program. Goodbye.')
            break

stream.stop_stream()
stream.close()
p.terminate()
cap.release()
