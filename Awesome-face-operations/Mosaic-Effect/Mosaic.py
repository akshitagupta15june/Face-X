import cv2
def do_mosaic (frame, x, y, w, h, neighbor=9):
  fh, fw=frame.shape [0], frame.shape [1]
  if (y + h>fh) or (x + w>fw):
    return
  for i in range (0, h-neighbor, neighbor):#keypoint 0 minus neightbour to prevent overflow
    for j in range (0, w-neighbor, neighbor):
      rect=[j + x, i + y, neighbor, neighbor]
      color=frame [i + y] [j + x] .tolist () #key point 1 tolist
      left_up=(rect [0], rect [1])
      right_down=(rect [0] + neighbor-1, rect [1] + neighbor-1) #keypoint 2 minus one pixel
      cv2.rectangle (frame, left_up, right_down, color, -1)
im=cv2.imread ("test.jpg", 1)
do_mosaic (im, 219, 61, 460-219, 412-61)
while 1:
  k=cv2.waitkey (10)
  if k == 27:
    break
  cv2.imshow ("mosaic", im)