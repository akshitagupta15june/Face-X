from PIL import Image

img_path=input("Enter your image path here:")  #ex -> r'C:\Users\xyz\images\p1.jpg' 
img = Image.open(img_path)  
img = img.resize((900, 900), Image.ANTIALIAS)
img.save('resized_image.jpg')

filename1 =  'resized_image.jpg'

filename=input("Literacy.png") 

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
background.save("International Literacy Day Filtered Image.png", format="png")
background.show()
