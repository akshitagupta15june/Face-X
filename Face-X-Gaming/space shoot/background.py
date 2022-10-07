import random
import pygame
import utils


class Background:
    def __init__(self):
        self.background = pygame.image.load("assets/sprites/stars-nebulae/blackspace-background.png").convert()

        self.stars_surfaces = [pygame.image.load("assets/sprites/stars-nebulae/stars.png").convert_alpha(),
                               pygame.image.load("assets/sprites/stars-nebulae/stars.png").convert_alpha()]

        self.stars = []
        for surface in self.stars_surfaces:
            self.stars.append([surface, surface.get_rect()])

        self.nebulae_surfaces = [
            pygame.image.load("assets/sprites/stars-nebulae/nebula1.png").convert_alpha(),
            pygame.image.load("assets/sprites/stars-nebulae/nebula2.png").convert_alpha(),
            pygame.image.load("assets/sprites/stars-nebulae/nebula3.png").convert_alpha()
        ]
        self.nebulae = []
        for surface in self.nebulae_surfaces:
            self.nebulae.append([surface, surface.get_rect()])

        self.stars_velocity = 8
        self.nebula_velocity = 2

    def draw_background(self, screen_surface):
        screen_surface.blit(self.background, (0, 0))
        self.draw_nebulae(screen_surface)
        self.draw_stars(screen_surface)

    def draw_nebulae(self, screen_surface):
        self.nebulae[0][1].top += self.nebula_velocity

        for index in range(0, len(self.nebulae)):
            screen_surface.blit(self.nebulae[index][0], self.nebulae[index][1])
            if index < len(self.nebulae) - 1:
                self.nebulae[index + 1][1].bottom = self.nebulae[index][1].top

        randomic_x = [0, 100, 200, 300]

        if self.nebulae[1][1].bottom >= utils.SCREEN_HEIGHT:
            self.nebulae[2][1].centerx = randomic_x[random.randint(0, 3)]
            aux = self.nebulae.pop(0)
            self.nebulae.append(aux)

    def draw_stars(self, screen_surface):
        self.stars[0][1].top += self.stars_velocity

        randomic_x = [0, 100, 200, 300]

        if self.stars[0][1].top >= utils.SCREEN_HEIGHT:
            self.stars[0][1].centerx = randomic_x[random.randint(0, 3)]
            aux = self.stars.pop(0)
            self.stars.append(aux)

        for index in range(0, len(self.stars)):
            screen_surface.blit(self.stars[index][0], self.stars[index][1])
            if index < len(self.stars) - 1:
                self.stars[index + 1][1].bottom = self.stars[index][1].top
