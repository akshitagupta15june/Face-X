import sys,time,random,pygame
from collections import deque
import cv2 as cv, mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1,circle_radius=1)
pygame.init()

# we use openCV to capture image using camera
VideoCapture = cv.VideoCapture(0)
# this tuple checks that all elements are with in the bound of the screen
window_size = (VideoCapture.get(cv.CAP_PROP_FRAME_WIDTH),VideoCapture.get(cv.CAP_PROP_FRAME_HEIGHT))
# we set up the pygame window
screen = pygame.display.set_mode(window_size)
#  bird and image variables
bird = pygame.image.load("bird_sprite.png")
bird = pygame.transform.scale(bird,(bird.get_width()/6,bird.get_height()/6))
bird_rect = bird.get_rect()
#  we set up the bird's position
bird_rect.center = (window_size[0]//6,window_size[1]//2)
pipe_frames = deque()
#  we set up the pipe's position
pipe_img = pygame.image.load("pipe_sprite_single.png")
pipe_starting_template = pipe_img.get_rect()
space_between_pipes = 250

#  Game variables
game_clock = time.time()
stage =1
pipeSpawnTimer = 0
time_between_pipe_spawn = 40
distance_between_pipes = 500
pipe_velocity = lambda: distance_between_pipes//time_between_pipe_spawn
score = 0
didUpdateGame = False
game_is_running = True


#  we make sure face reccognition works
with mp_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5) as face_mesh:
    while True:
        if not game_is_running:
            text = pygame.font.SysFont("Helvetica", 50).render("Game Over", True, (99,248, 255))
            tr = text.get_rect()
            tr.center = (window_size[0]//2,window_size[1]//2)
            screen.blit(text,tr)
            pygame.display.update()
            pygame.time.wait(5000)
            VideoCapture.release()
            cv.destroyAllWindows()
            pygame.quit()
            sys.exit()


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        # we capture the image from the camera
        ret,frame = VideoCapture.read()
        # we convert the image to RGB
        if not ret:
            print("Empty frame")
            continue

        screen.fill((255,255,255))
        frame.flags.writeable = False
        frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        results = face_mesh.process(frame)
        frame.flags.writeable = True
        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0 :
            marker = results.multi_face_landmarks[0].landmark[94].y
            bird_rect.centery = (marker - 0.5) * 1.5 * window_size[1] + 0.5 * window_size[1]
            if bird_rect.top < 0:
                bird_rect.y = 0
            if bird_rect.bottom > window_size[1]:
                bird_rect.y = window_size[1] - bird_rect.height
        frame = cv.flip(frame,1).swapaxes(0,1)


        for pf in pipe_frames:
            pf[0].x -= pipe_velocity()
            pf[1].x -= pipe_velocity()
            
        if len(pipe_frames) > 0 and pipe_frames[0][0].right < 0:
            pipe_frames.popleft()

        

        # we convert the image to pygame format
        pygame.surfarray.blit_array(screen,frame)
        screen.blit(bird,bird_rect)
        checker = True
        for pf in pipe_frames:
            if pf[0].left <= bird_rect.x <= pf[0].right:
                checker = False
                if not didUpdateGame:
                    score += 1
                    didUpdateGame = True
                screen.blit(pipe_img,pf[1])
                screen.blit(pygame.transform.flip(pipe_img,0,1),pf[0])
        if checker:
            didUpdateGame = False 
        
       # Stage, score text
        text = pygame.font.SysFont("Helvetica Bold.ttf", 50).render(f'Stage {stage}', True, (99, 245, 255))
        tr = text.get_rect()
        tr.center = (100, 50)
        screen.blit(text, tr)
        text = pygame.font.SysFont("Helvetica Bold.ttf", 50).render(f'Score: {score}', True, (99, 245, 255))
        tr = text.get_rect()
        tr.center = (100, 100)
        screen.blit(text, tr)

        # Update screen
        pygame.display.flip()

        # Check if bird is touching a pipe
        if any([bird_rect.colliderect(pf[0]) or bird_rect.colliderect(pf[1]) for pf in pipe_frames]):
            game_is_running = False

        # Time to add new pipes
        if pipeSpawnTimer == 0:
            top = pipe_starting_template.copy()
            top.x, top.y = window_size[0], random.randint(120 - 1000, window_size[1] - 1000 -space_between_pipes)
            bottom = pipe_starting_template.copy()
            bottom.x, bottom.y = window_size[0], top.y + 1000 + space_between_pipes
            pipe_frames.append([top, bottom])

        # Update pipe spawn timer - make it cyclical
        pipeSpawnTimer += 1
        if pipeSpawnTimer >= time_between_pipe_spawn: pipeSpawnTimer = 0

        # Update stage
        if time.time() - game_clock >= 10:
            time_between_pipe_spawn *= 5 / 6
            stage += 1
            game_clock = time.time()