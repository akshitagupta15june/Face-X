import cv2

def cartoonify_image(image_path):
    image = cv2.imread(image_path)
    
    height, width, _ = image.shape
    resized_image = cv2.resize(image, (600, 450))
    
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray_image, 9)
    
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 11)
    original_image = cv2.bilateralFilter(resized_image, 9, 300, 300)

    cartoon = cv2.bitwise_and(original_image, original_image, mask=thresh)
    cartoon = cv2.resize(cartoon, (width, height))
    
    return cartoon
