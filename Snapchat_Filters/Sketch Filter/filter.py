import cv2
import numpy as np

def sketch_filter(image, sigma_s=50, sigma_r=0.3, blend_strength=0.4):
    #Convert to gray
    img_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    
    #Invert 
    img_blur = cv2.GaussianBlur(img_gray, (21,21), 0, 0)
    
    #Blend
    sketch_img = cv2.divide(img_gray, img_blur, scale=256)
    
    #Paint on canvas
    canvas_img = cv2.multiply(sketch_img, bg, scale=1./256)
    
    #Reduce strength of canvas  
    blend_img = cv2.addWeighted(sketch_img, 0.35 , canvas_img, 0.65, 0)
    blend_img = cv2.cvtColor(blend_img, cv2.COLOR_GRAY2RGB)
    
    #Soften colors in image
    color_img = cv2.bilateralFilter(image, 10, 100, 200)
    
    bs = .95
    final_img = cv2.addWeighted(blend_img, bs , color_img, 1 - bs, 0)
    
    return final_img

#Start video capture from webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

bg = cv2.imread("gray_bg.jpg",cv2.CV_8UC1)
bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]))

while True:
    ret, frame = cap.read()
    
    #Display final result
    cv2.imshow("Cartoon effect", sketch_filter(frame))
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()