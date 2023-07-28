import cv2
import numpy as np
import random
from scipy.spatial import Voronoi, voronoi_plot_2d

def generate_random_points(img_shape, num_points):
    return [(random.randint(0, img_shape[1]), random.randint(0, img_shape[0])) for _ in range(num_points)]

def mosaic(img, num_points):
    # Get the image size
    height, width, _ = img.shape

    # Generate random points for the Voronoi diagram
    points = generate_random_points((height, width), num_points)

    # Create a Voronoi diagram from the random points
    vor = Voronoi(points)

    # Create an empty canvas for the Voronoi pattern
    output_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Generate the Voronoi pattern and apply the average color to each region
    old_col=0
    for region_index in range(len(vor.regions) - 1):
        region = vor.regions[region_index]
        if -1 not in region and len(region) > 0:
            # Get the coordinates of the vertices of the current region
            vertices = [vor.vertices[i] for i in region]

            # Create a mask for the current Voronoi region
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(vertices, np.int32).reshape((-1, 1, 2))], color=255)

            # Calculate the average color of the region defined by the mask
            avg_color = cv2.mean(img, mask=mask)[:3]
            old_col = avg_color

            # Fill the Voronoi region with the average color
            output_img[mask == 255] = avg_color

    return output_img


#Start video capture from webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

while True:
    ret, frame = cap.read()
    
    #Display final result
    cv2.imshow("Cartoon effect", mosaic(frame,1000))
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()