from PIL import Image

img = Image.open('p8.jpg')   #image name input (exp-> "xyz.jpg")

img = img.resize((900, 900), Image.ANTIALIAS)
img.save('resized_image.jpg')
#img.show()


filename1 = 'resized_image.jpg'  
  
# Back Image
filename = 'b4.png'  #frame image already given 
  
# Open Front Image
frontImage = Image.open(filename)
  
# Open Background Image
background = Image.open(filename1)
  
# Convert image to RGBA
frontImage = frontImage.convert("RGBA")
  
# Convert image to RGBA
background = background.convert("RGBA")


# Calculate width to be at the center
width = (background.width - frontImage.width) // 2
  
# Calculate height to be at the center
height = (background.height - frontImage.height) // 2
  
# Paste the frontImage at (width, height)
background.paste(frontImage, (width, height), frontImage)
  
# Save this image
background.save("Filter_frame.png", format="png")
background.show()
