from PIL import Image

from PIL import ImageFilter

 

# Load the image

img = Image.open("faceneg.png");

 

# Display the original image

img.show()

 

# Read pixels and apply negative transformation

for i in range(0, img.size[0]-1):

    for j in range(0, img.size[1]-1):

        # Get pixel value at (x,y) position of the image

        pixelColorVals = img.getpixel((i,j));

       

        # Invert color

        redPixel    = 255 - pixelColorVals[0]; # Negate red pixel

        greenPixel  = 255 - pixelColorVals[1]; # Negate green pixel

        bluePixel   = 255 - pixelColorVals[2]; # Negate blue pixel

                   

        # Modify the image with the inverted pixel values

        img.putpixel((i,j),(redPixel, greenPixel, bluePixel));

 

# Display the negative image

img.show();