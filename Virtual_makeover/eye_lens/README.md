# EYE LENS COLOR
The eye lens color can changed be using facial landmark detection (shape_predictor_70_face_landmarks.dat).

# APPROACH
The centres of two eyes are found, which are points 68 and 69 in shape predictor. A mask of eyes as white color is created on a black blackground. The color of the lens is changed using trackbar which is reflected on the mask. Finally, the mask and original image are combined using bitwise or operation.

# STEPS

![Mask](./eye lens/mask.png)

# OUTPUTS


