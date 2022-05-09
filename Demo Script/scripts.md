# Demo Narration Scripts

## Zijie Liu

Hi, my names is Zijie Liu.

## Mingyang Gao

And my name is Mingyang Gao.

## Zijie Liu

This video is the demo of our project. Our topic is digital theremin.

This is a photo of a theremin. It is an instrument with a quite futuristic look. It has two metal antennas and it can use the antennas to detect positions and gestures of hands, and produce a sound with certain gain and frequency according to the position and gestures detected. In order to reproduce this mechanism, we designed a program, with a process much alike.

To detect hands, we used a Python package called mediapipe. It is developed by Google, and we implemented its hand detection process. Also, part of the codes are inspired by Nicholas Renotte, much thanks to his public tutorial videos and codes. (Github page and YouTube page here)

Although a part of the codes are inspired by Nicholas Renotte, by "inspired" I mean that I only learnt the thoughts implemented inside his tutorial and the detection codes and custom functions I used are designed by myself. (i.e. The functions in myfunctions.py are designed by myself and tested usable, except for the `clip16` function, which comes from the demo) Also, the overall hand detection codes come from the Google mediapipe ofiicial demo, link on the screen.

After running the program, as you can see, the window pops up and it shows the size of the image, which hands controls what parameter, the parameter values and the output status. The program will only output the sound when both hands are detected. But because mediapipe uses a neural network to detect hands, it may recognize something else as a hand by mistake. But those are only occasional cases and can be corrected very fast.

This is a picture of the recognition points on a hand. After two hands are correctly recognized, it detects some points and calculate the parameters.

For the gain, It takes the five points at the edge of the palm of left hand (without the wrist point), calculate its average point, take the vertical position of this point as a weight, and apply the weight to maximum gain (32767, or $2^{15}-1$) to get final gain.

As for the frequency, it takes the five points at the tips of the fingers, calculate the average distance between them, and use this as a weight to apply to the range of frequency change (in this case, the range is from 200 Hz to 5120 Hz). Also, we applied a logarithmic change, which can make the pitch change sound linear.

## Mingyang Gao

After getting the parameters, we need to implement the sound control process. This part is inspired by the demo No.17 in class. But instead of using a predefined GUI, we recognize hands and use our hands to control the parameters. When there is no hand or only one hand is detected, there will be no output, the parameters and the output block will be set to zero. When two hands are detected, it will output a sound with the designated volume (or gain) and pitch (or frequency).

On top of that, we implemented a linear change to the gain and the frequency, so that there will be no audio artifact when the parameters change drastically.

Finally, to stop the program, simply press button 'Q' and the program will stop. The program detects whether the button is pressed at the end of each loop.

## Zijie Liu

That is all for our project.

## Mingyang Gao

And thank you for watching.