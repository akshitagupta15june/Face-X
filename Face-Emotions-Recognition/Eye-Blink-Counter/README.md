# Eye Blinking detection

## When is the eye blinking?

If you think about this question, what possible explanation could you give to describe the blinking of the eyes? Probably more then one answer will be right.

Let’s do some brainstorming to define an eye that is blinking.

<h4>An eye is blinking when:</h4>
<ul>
<li>The eyelid is closed
<li>We can’t see the eyeball anymore
<li>Bottom and upper eyelashes connect together
</ul>

And also we need to take into account that all this actions must happen for a short amount of time (approximately a blink of an eye takes 0.3 to 0.4 seconds) otherwise it meas that the eye is just closed.

Now that we have found some possible answer to detect the blinking of the eye, we should focus on what’s possible to detect using Opencv, possibly choosing the easiest and most reliable solution with what we already have.

## Detecting the blinking

This is how the lines look like when the eye is open.
<img src="https://github.com/akshitagupta15june/Face-X/blob/master/Face-Emotions-Recognition/Eye-Blink-Counter/images/eye_open.jpg" align="centre">

This when the eye is closed.
<img src="https://github.com/akshitagupta15june/Face-X/blob/master/Face-Emotions-Recognition/Eye-Blink-Counter/images/eye_closed.jpg" align="centre">

<h4>What can you notice?</h4>

We can clearly see that the size of the horizontal line is almost identical in the closed eye and in the open eye while the vertical line is much longer in the open eye in coparison with the closed eye.
In the closed eye, the vertical line almost disappears.

We will take then the horizontal line as the point of reference, and from this we calculate the ratio in comparison with the vertical line.
If the the ratio goes below a certain number we will consider the eye to be closed, otherwise open.

On python then we create a function to detect the blinking ratio where we insert the eye points and the facial landmark coordinates and we will get the ratio between these two lines.

```bash
def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio
```

We will then use the ratio number later to detect and we can finally define when the eye is blinking or not.
In this case I found ratio number 5.7 to be the most reiable threshold, at least for my eye.

```bash
landmarks = predictor(gray, face)
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
        if blinking_ratio > 5.7:
            cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0))
```

<img src="https://github.com/akshitagupta15june/Face-X/blob/master/Face-Emotions-Recognition/Eye-Blink-Counter/images/working.png" align="centre">
