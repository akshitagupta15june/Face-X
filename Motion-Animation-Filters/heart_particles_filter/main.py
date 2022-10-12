import typing
import random
import numpy as np
import cv2 as cv
import time


# Initialize the camera
cap = cv.VideoCapture(0)

class HeartParticle:
	def __init__(self, start_pos, particle_speed_x, particle_speed_y, scale, fade_pos) -> None:
		# Particle start position
		self.pos = start_pos
		# Image scale
		self.scale = scale
		# Particle horizontal speed (Technically it is not used in this version but if you would like to make more complex movement then it would be needed)
		self.particle_speed_x = particle_speed_x
		# Particle vertical speed
		self.particle_speed_y = particle_speed_y
		# Load particle image
		self.particle = self.load_image()
		# Position after crossing which particle starts to fade
		self.fade_pos = fade_pos
		# Current particle alpha(transparency)
		self.alpha = 0.7


	def load_image(self) -> np.array:
		try:
			# We load the image and resize it according to the scale
			return cv.resize(cv.imread('heart_particle.png'), (int(100 * self.scale), int(100 * self.scale)))		
		except:
			print('No image was found')
			exit(1)

	def move(self):
		# Move particle
		self.pos[0] += self.particle_speed_x
		self.pos[1] += self.particle_speed_y

	def is_alpha(self) -> bool:
		# Check if alpha is 0
		if self.alpha <= 0.0:
			return False
		return True


	# It is a separate function for better performance
	def update_state(self):
		# Check if particle crossed the fading position
		if self.pos[1] < self.fade_pos:
			self.alpha -= 0.1

	def draw_particle_portion(self, current_frame, start_pos, end_pos) -> np.array:
		# Particle width
		foreground_width = self.particle.shape[1]
		# Particle height
		foreground_height = self.particle.shape[0]
		# Particle portion
		blended_portion = np.zeros((foreground_height, foreground_width, 3), dtype=np.uint8)

		# Set coordinates
		start_x, start_y = start_pos
		end_x, end_y = end_pos
		

		# Iterate through y image coordinate
		for y in range(start_y, end_y):
			# Iterate through x image coordinate
			for x in range(start_x, end_x):
				# Check if current pixel color not equals to (80, 25, 226)
				if (self.particle[y - start_y, x - start_x] != (80, 25, 226)).all(): # Confusing statement
					# set color values to current frame's current pixel 
					blue =  current_frame[y, x, 0]
					green = current_frame[y, x, 1] 
					red =   current_frame[y, x, 2]

					# Set the current pixel's color of particle portion
					blended_portion[y - start_y, x  - start_x] = (blue, green, red)
				else:
					# Blending formula: a * alpha + b * beta + gamma
					# beta = (1 - alpha)
					# Blend color values of current frame and particle image
					blue =  (self.particle[y - start_y, x - start_x, 0] * self.alpha) + (current_frame[y, x, 0] * (1 - self.alpha))
					green = (self.particle[y - start_y, x - start_x, 1] * self.alpha) + (current_frame[y, x, 1] * (1 - self.alpha))
					red =   (self.particle[y - start_y, x - start_x, 2] * self.alpha) + (current_frame[y, x, 2] * (1 - self.alpha))
					# Set the current pixel's color of particle portion
					blended_portion[y - start_y, x  - start_x] = (blue, green, red)

		return blended_portion
		

	def blend_in(self, background: np.array) -> np.array:
		# Frame Height
		background_height = background.shape[0]
		# Frame Width
		background_width = background.shape[1]
		# Particle width
		foreground_width = self.particle.shape[1]
		# Particle height
		foreground_height = self.particle.shape[0]


		# Check if particle reached top boundary
		if self.pos[1] <= 0:
			self.alpha = 0
			self.pos[1] = 0

		# Image coordinates
		start_x, start_y = self.pos

		# Image end coordinates
		end_y = self.pos[1] + foreground_height
		end_x = self.pos[0] + foreground_width
		
		# Create particle portion
		blended_portion = self.draw_particle_portion(background, self.pos, (end_x, end_y))

		# We replace old pixels with new added ones
		background[start_y:end_y, start_x:end_x, :] = blended_portion
		return background


class Blender:
	def __init__(self, init_frame=None):
		# Initialize the fram
		self.frame = init_frame
		# Set window width
		self.width = 640
		# Set window height
		self.height = 480
		# Those variables were created for particles to be able to go beyond the window boundaries
		# self.padding_x = 0
		# self.padding_y = 0

		#{particle: [False(moved?), False(blended?)]}
		self.particles = {}


	def set_frame(self, frame):
		# Set the current frame
		self.frame = frame

	def add_image(self, image) -> np.array:
		# Add the image to the dict
		self.particles[image] = [False, False]
		# Update the blender
		self.update()

	def remove_image(self, image):
		# If image is in blender
		if image in self.particles:
			# Remove image 
			del self.particles[image]
		else:
			print('No such element in blender.')

	def get_image_number(self) -> int:
		# Get number of images
		return len(self.particles)

	def update(self) -> np.array:
		# Array of particles to be removed
		to_remove_obj = []

		# Iterate through particles and their flags
		for i, flags in self.particles.items():
			# Check if particle has alpha lesser than 0 (so that means the particle faded)
			if not i.is_alpha():
				# We can't remove here because it will throw an exception that dictionary size changed during runtime
				to_remove_obj.append(i)
				continue
			i.update_state()
			
			# Check if particle hasn't moved yet
			if not flags[0]:
				# Move particle
				i.move()
				# Set moved flag to True
				self.particles[i][0] = True

			# Check if particle hasn't been blended yet
			if not flags[1]:
				# Blend particle with current frame and override current frame with new frame
				self.frame = i.blend_in(self.frame)
				# Set blended flag to True
				self.particles[i][1] = True
					

		for i in to_remove_obj:
			self.remove_image(i)

		# print('Speed: x({}) y({}), Pos: x({}) y({})'.format(i.particle_speed_x, i.particle_speed_y, i.pos[0], i.pos[1]))
		# print('Flags: 0({}) 1({})'.format(flags[0], flags[1]))

	def get_frame(self):
		# Return blender frame
		return self.frame

	def reset(self):
		# Reset all the flags
		for i, flags in self.particles.items():
			flags[0] = False
			flags[1] = False



# TODO: You can improve this by adding the fullscreen spawn mode  

# Blender instance
blender = Blender()

# Import face-detection model
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')


# Particles minimum and maximum vertical speeds
particle_min_speed_y = -12
particle_max_speed_y = -8
# Size variations
particle_sizes = [0.1, 0.15, 0.2]
# Limit of particles on the screen
limit = 35
# Spawn rate
speed = 3
max_number = limit
# Application's frame rate
frame_rate = 60
prev = 0

while True:
	time_elapsed = time.time() - prev
	_, frame = cap.read()

	if time_elapsed > 1. / frame_rate:
		prev = time.time()
		# Set blender frame
		blender.set_frame(frame)
		# Update blender
		blender.update()
		# Convert frame to gray scale
		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		# Get faces coordinates and sizes
		faces = face_cascade.detectMultiScale(gray, 1.1, 4)
		# Pos after crossing which heart particle fades
		fade_pos = []
		for (x, y, w, h) in faces:
			fade_pos.append(y)

		# Set limit number of particles depending on amount of faces detected
		max_number = limit * len(faces)


		# Iterate through all faces
		for i in range(max_number // limit):
			# If current amount of particles on the screen is lesser than limit number
			if blender.get_image_number() < max_number:
				# Spawn rate
				for j in range(speed):
					# Randomize position of particle in face ranges
					pos = [random.randint(faces[i][0], faces[i][0] + faces[i][2]), random.randint(faces[i][1], faces[i][1] + faces[i][3])]
					# Randomize the size of the particle
					size = random.choice(particle_sizes)
					# Check if randomized values are not out of the boundaries
					if pos[0] + int(size * 100) < frame.shape[1] and pos[1] + int(size * 100) < frame.shape[0]: 
						# Add particle to blender
						blender.add_image(HeartParticle(pos, 0, random.randint(particle_min_speed_y, particle_max_speed_y), size, faces[i][1]))

		# Reset blender
		blender.reset()

		# Show the blender frame
		cv.imshow('img', blender.get_frame())

	

	if cv.waitKey(1) == ord('q'):
		break


cap.release()
cv.destroyAllWindows()



