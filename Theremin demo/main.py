import mediapipe as mp
import cv2
import pyaudio, struct
import numpy as np
from numpy import pi

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

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
  # specify low frames_per_buffer to reduce latency

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

    # declare some paremeters for text printing
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_size = 0.5
    font_thickness = 1

    # print info about input video feed on the screen
    h, w, c = image.shape
    color_imgsize = (0, 0, 255)
    font_pos_imgsize = (10, 25)
    image = cv2.putText(image, 'Input image height: {}, width: {}'.format(h, w), font_pos_imgsize, font, font_size, color_imgsize, font_thickness)

    # Print the image
    cv2.imshow('Digital Theremin', image)

    # wait for key 'q' to be pressed
    # when pressed, quit
    if cv2.waitKey(5) & 0xFF == ord('q'):
      break

stream.stop_stream()
stream.close()
p.terminate()
cap.release()