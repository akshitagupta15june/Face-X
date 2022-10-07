"""
*
* (Pygame, Mediapipe, OpenCV) An adaptation of the SPACE SOLDIER game that can be controlled 
*  with movements of the face, having the tip of the nose as a pivot.
*
* Sprites: https://opengameart.org/content/barrier-frontier-spaceships
*          https://opengameart.org/content/space-ship-construction-kit
*          https://opengameart.org/content/sci-fi-space-simple-bullets
*          https://opengameart.org/content/explosion
*          Autor
* Sounds: https://opengameart.org/content/retro-shooter-sound-effects
* Fonts: https://www.dafont.com/pt/alien-eclipse.font
"""

import pygame
from pygame.time import Clock
from pygame.locals import *
from background import Background
from enemies.enemies_manager import EnemiesManager
from player.player_manager import PlayerManager
from target import Target
import sys
import utils
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

BLUE = (255, 0, 0, 0)
GREEN = (0, 255, 0, 0)
RED = (0, 0, 255, 0)
NOSE = [4]

TOP_LEFT_X = 220
TOP_LEFT_Y = 50
BOTTOM_RIGHT_X = int(utils.SCREEN_WIDTH/2) + TOP_LEFT_X
BOTTOM_RIGHT_Y = int(utils.SCREEN_HEIGHT/2) + TOP_LEFT_Y

pygame.init()
pygame.mixer.init()
FPS = 24
clock = Clock()
SCREEN_SURFACE = pygame.display.set_mode((utils.SCREEN_WIDTH, utils.SCREEN_HEIGHT))
pygame.display.set_caption("Space Shoot")

class Game:

    def __init__(self):
        self.background = Background()
        self.player_manager = PlayerManager()
        self.enemies_manager = EnemiesManager()
        self.cont_frame = 0
        self.target = Target()
        self.mouse_on_screen = False
        self.start = False
        self.title_part_a = "Space"
        self.title_part_b = "Shoot"
        self.font_title = pygame.font.Font("assets/fonts/Alien-Eclipse.ttf", 90)
        self.font_score_on_pause = pygame.font.Font("assets/fonts/Alien-Eclipse.ttf", 40)
        self.font_score = pygame.font.Font("assets/fonts/Alien-Eclipse.ttf", 24)
        self.info_autor_font = pygame.font.Font("assets/fonts/Alien-Eclipse.ttf", 12)
        self.draw_title_first_time = True
        self.player_x = 0
        self.player_y = 0
        self.acumulate_score = 0

    def draw_title(self):
        if not self.draw_title_first_time:
            score_text = "Scores "+str(self.player_manager.score)
            score_surface = self.font_score_on_pause.render(score_text, True, (255, 165, 0))
            x = int(utils.SCREEN_WIDTH / len(score_text))
            SCREEN_SURFACE.blit(score_surface, (x, 300))
        
        autor_info_text = "By Rishyanth"
        autor_info_text_surface = self.info_autor_font.render(autor_info_text, True, (255, 165, 0))
        SCREEN_SURFACE.blit(autor_info_text_surface, (5, 20))

        autor_info_text = "https://github.com/Rishyanth"
        autor_info_text_surface = self.info_autor_font.render(autor_info_text, True, (255, 165, 0))
        SCREEN_SURFACE.blit(autor_info_text_surface, (5, 45))

        title_surface = self.font_title.render(self.title_part_a, True, (255, 69, 0))
        SCREEN_SURFACE.blit(title_surface, (65, 100))

        title_surface = self.font_title.render(self.title_part_b, True, (255, 69, 0))
        SCREEN_SURFACE.blit(title_surface, (20, 200))

    def draw_score(self, score):
        score_text = str(score)
        score_surface = self.font_score.render(score_text, True, (255, 165, 0))
        SCREEN_SURFACE.blit(score_surface, (20, 20))

    def start_game(self):
        self.enemies_manager.create(self.cont_frame)
        self.enemies_manager.fire(SCREEN_SURFACE, self.cont_frame)
        self.enemies_manager.update(SCREEN_SURFACE, self.cont_frame)

    def palse_game(self):
        while len(self.player_manager.player_group) == 0:
            self.enemies_manager.enemy_group.empty()
            self.enemies_manager.bullet_group.empty()
            self.target.target_on_player = False
            self.player_manager.create(200, 550)

    def draw_player(self, x, y):
        self.player_manager.update(SCREEN_SURFACE, x, y, self.target)
        self.player_manager.fire(SCREEN_SURFACE, self.cont_frame)

    def run_game_logic(self, controller_pos):
        
        self.cont_frame += 1
        if self.cont_frame == 100:
            self.cont_frame = 0

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        if controller_pos["x"] <= 0 or controller_pos["x"] >= utils.SCREEN_WIDTH - 5 or \
            controller_pos["y"] <= 0 or controller_pos["y"] >= utils.SCREEN_HEIGHT - 5:
                self.mouse_on_screen = False
        else:
            self.mouse_on_screen = True
            self.player_x = controller_pos["x"]
            self.player_y = controller_pos["y"]

        self.background.draw_background(SCREEN_SURFACE)

        if self.target.target_on_player:
            self.draw_title_first_time = False
            self.start = True
        if self.start:
            self.start_game()
            self.draw_score(self.acumulate_score)

        self.acumulate_score += self.enemies_manager.destroy(SCREEN_SURFACE, self.player_manager.bullet_group)
       
        enemies = self.enemies_manager.enemy_group
        enemy_bullets = self.enemies_manager.bullet_group

        if self.player_manager.destroy(enemies, enemy_bullets):
            self.start = False
            self.player_manager.score = self.acumulate_score
            self.acumulate_score = 0

        if not self.start:
            self.palse_game()
            self.draw_title()

        self.draw_player(self.player_x, self.player_y)

        player_rect = self.player_manager.player.rect
        if self.mouse_on_screen:
            self.target.update(SCREEN_SURFACE, self.player_x, self.player_y, player_rect)

        else:
            self.target.update(SCREEN_SURFACE, -100, -100, player_rect)

        pygame.display.update()
        # clock.tick(FPS)

    def get_landmark_pos(self, face_mesh_results_multi_face_landmarks, flm_list, image):
        h, w, c = image.shape

        for face_landmarks in face_mesh_results_multi_face_landmarks:
            for idd, lm in enumerate(face_landmarks.landmark):
                if idd in flm_list:
                    x, y = int(lm.x * w), int(lm.y * h)
                    pos = {"idd": idd, "x": x, "y": y}
                    return pos
        
    def convert_landmark_pos_to_controler_pos(self, pos):
        controller_pos = {}
        controller_pos["x"] = (BOTTOM_RIGHT_X - pos['x']) * 2
        controller_pos["y"] = (pos['y'] - TOP_LEFT_Y) * 2
        return controller_pos
    
    def draw_landmark(self, pos, image):
        cv2.rectangle(image, (TOP_LEFT_X, TOP_LEFT_Y), 
                (BOTTOM_RIGHT_X, BOTTOM_RIGHT_Y), 
                BLUE, 2)

        if pos["x"] >= TOP_LEFT_X and pos["x"] <= int(utils.SCREEN_WIDTH/2)+TOP_LEFT_X and \
            pos["y"] >= TOP_LEFT_Y and pos["y"] <= int(utils.SCREEN_HEIGHT/2)+TOP_LEFT_Y:
            cv2.circle(image, (pos["x"], pos["y"]), 10, GREEN, cv2.FILLED, 2)
        
        else:
            cv2.circle(image, (pos["x"], pos["y"]), 10, RED, cv2.FILLED, 2)

    def run_mediapipe(self):
        
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
            
            while cap.isOpened():
                success, image = cap.read()

                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    break

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)

                # Draw the face mesh annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_face_landmarks:
                    pos = self.get_landmark_pos(results.multi_face_landmarks, NOSE, image)
                    controller_pos = self.convert_landmark_pos_to_controler_pos(pos)
                    self.draw_landmark(pos, image)
                
                self.run_game_logic(controller_pos)
                
                cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            cap.release()


if __name__ == "__main__":
    Game().run_mediapipe()
