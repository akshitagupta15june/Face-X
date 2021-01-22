"""Generate image from array of numbers.

(Script not mentioned in tutorial)
"""

import numpy as np
import cv2

string = """0 0 0 0 0 0 0
0 0 0 1 0 0 0
0 0 1 1 1 0 0
0 1 1 1 1 1 0
0 0 1 1 1 0 0
0 0 0 1 0 0 0
0 0 0 0 0 0 0"""

array = np.array([
    list(map(int, line.split())) for line in string.split('\n')
]) * 255.

larger = cv2.resize(array, None, fx=60, fy=60, interpolation=cv2.INTER_AREA)

cv2.imwrite('outputs/diamond.png', larger)

string2 = """1  1  1  1  1  1  1  1  1  1  1  1
1  1  1  1  0  0  0  0  1  1  1  1
1  1  0  0 .4 .4 .4 .4  0  0  1  1
1  0 .4 .4 .5 .4 .4 .4 .4 .4  0  1
1  0 .4 .5 .5 .5 .4 .4 .4 .4  0  1
0 .4 .4 .4 .5 .4 .4 .4 .4 .4 .4  0
0 .4 .4 .4 .4  0  0 .4 .4 .4 .4  0
0  0 .4 .4  0  1 .7  0 .4 .4  0  0
0  1  0  0  0 .7 .7  0  0  0  1  0
1  0  1  1  1  0  0 .7 .7 .4  0  1
1  0 .7  1  1  1 .7 .7 .7 .7  0  1
1  1  0  0 .7 .7 .7 .7  0  0  1  1
1  1  1  1  0  0  0  0  1  1  1  1
1  1  1  1  1  1  1  1  1  1  1  1""".replace(' ', ' ')

array = np.array([
    list(map(float, line.split())) for line in string2.split('\n')
]) * 255.

larger = cv2.resize(array, None, fx=60, fy=60, interpolation=cv2.INTER_AREA)

cv2.imwrite('outputs/pokeball.png', larger)
