import mediapipe as mp
import cv2
import pyaudio, struct
import numpy as np
from numpy import pi

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

# Create Pyaudio object
p = pyaudio.PyAudio()
stream = p.open(
  format = pyaudio.paInt16,  
  channels = 1, 
  rate = Fs,
  input = False, 
  output = True,
  frames_per_buffer = 128)

# specify output block related parameters

# for an output without artifact, block length cannot be smaller than 512
BLOCKLEN = 512

output_block = [0] * BLOCKLEN
theta = 0
f1 = 500    # temporary value for f1
gain = 12800    # temp value for gain

# Capture WebCam feed
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()

    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

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

    # Flip the image horizontally for a selfie-view display.
    image = cv2.flip(image, 1)

    # Draw zones for recognizing both hands
    # Zone 1 (Left hand, controls gain)
    color_zone1 = (255, 0, 0)
    rect_zone1_pos1 = (40, 40)
    rect_zone1_pos2 = (300, 440)
    txt_pos_zone1 = (rect_zone1_pos1[0] + 10, rect_zone1_pos1[1] + 25)
    cv2.rectangle(image, rect_zone1_pos1, rect_zone1_pos2, color_zone1, 2)
    cv2.putText(image, 'Zone 1 (Gain)', txt_pos_zone1, font, font_size, color_zone1, font_thickness)

    # Zone 2 (Right hand, controls frequency)
    color_zone2 = (255, 0, 255)
    rect_zone2_pos1 = (340, 150)
    rect_zone2_pos2 = (600, 440)
    txt_pos_zone2 = (rect_zone2_pos1[0] + 10, rect_zone2_pos1[1] - 20)
    cv2.rectangle(image, rect_zone2_pos1, rect_zone2_pos2, color_zone2, 2)
    cv2.putText(image, 'Zone 2 (Freq)', txt_pos_zone2, font, font_size, color_zone2, font_thickness)


    # print info about input video feed on the screen
    h, w, c = image.shape
    txt_color_imgsize = (0, 0, 255)
    txt_pos_imgsize = (10, 25)
    image = cv2.putText(image, 'Input image height: {}, width: {}'.format(h, w), txt_pos_imgsize, font, font_size, txt_color_imgsize, font_thickness)

    # Print the image
    cv2.imshow('Digital Theremin', image)

    # sound part
    om1 = 2.0 * pi * f1 / Fs

    for i in range(0, BLOCKLEN): 
        output_block[i] = int( gain * np.cos(theta) )
        theta = theta + om1

    # Reset theta when out of bound of pi
    if theta > pi:
        theta = theta - 2.0 * pi

    binary_data = struct.pack('h' * BLOCKLEN, *output_block)   # 'h' for 16 bits
    stream.write(binary_data)

    # Wait for key 'q' to be pressed
    # When pressed, quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

stream.stop_stream()
stream.close()
p.terminate()
cap.release()