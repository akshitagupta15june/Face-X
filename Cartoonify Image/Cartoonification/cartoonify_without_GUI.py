import cv2
import argparse

video_capture = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('cartoonised.avi', fourcc, 20.0, (1200, 600))

while (video_capture.isOpened()):
    ret, frame = video_capture.read()

    if ret == True:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_image= cv2.resize(gray, (1200, 600))
        blurred = cv2.medianBlur(resized_image, 9)

        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_MASK, 11,11)


        original_image = cv2.bilateralFilter(frame,9, 300, 300)

        cartoon = cv2.bitwise_and(original_image, original_image, mask= thresh)

        out.write(cartoon)



        cv2.imshow('Cartoon_image', cartoon)
        cv2.imshow('Original Image', frame)

        if cv2.waitKey(1) & 0xFF ==27:
            break

    else:
        print("Camera not available, Please upload a photo")


if(video_capture.isOpened() == False):
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-i", "--image", required=True, help= "Image Path")

    args= vars(arg_parse.parse_args())
    image = cv2.imread(args['image'])
    filename = 'Cartoonified_image.jpg'
    resized_image = cv2.resize(image, (600, 450))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.medianBlur(gray_image, 9)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11,11)

    original_image = cv2.bilateralFilter(image, 9, 300, 300)

    cartoon = cv2.bitwise_and(original_image, original_image, mask=thresh)
    cartoon_resize= cv2.resize(cartoon, (600,450))

    cv2.imshow("Cartoonified", cartoon_resize)
    cv2.imwrite(filename, cartoon)
    cv2.imshow("Main Image", resized_image)

cv2.waitKey(0)

out.release()
video_capture.release()
cv2.destroyAllWindows()








