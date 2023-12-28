import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

brush_radius = 5
intensity = 3

def oil_effect(image, brush_radius, intensity):
    # Apply the oil painting effect to the image
    oil_painting = cv2.xphoto.oilPainting(image,brush_radius, intensity)

    return oil_painting

def combine_img(img,bg,threshold=0.5):
    
    #Load image segmentation model
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

    #Create a mask for person and background
    results = selfie_segmentation.process(img)
    segmentation_mask = results.segmentation_mask

    #Process mask using threshold value
    condition = np.stack((segmentation_mask,) * 3, axis=-1)  > threshold

    #Use the mask to apply background to frame
    output_image = np.where(condition, img, bg)
    
    return output_image

def main():
    img = cv2.imread("input.png")
    
    oil_img = oil_effect(img, brush_radius, intensity)
    combine = cv2.cvtColor(combine_img(oil_img,img), cv2.COLOR_BGR2RGB)

    result = Image.fromarray(combine)

    result.save("oil_effect.png")

    print("Saved the result!")

if __name__ == '__main__':
    main()