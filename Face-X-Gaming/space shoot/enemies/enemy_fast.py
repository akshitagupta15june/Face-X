import pygame
import random
import utils
from .enemy_fast_bullet import EnemyFastBullet


class EnemyFast(pygame.sprite.Sprite):
    def __init__(self, midbottom_x, midbottom_y):
        super(EnemyFast, self).__init__()
        self.image = pygame.image.load("assets/sprites/space-ships/enemy-speed.png").convert_alpha()
        self.rect = self.image.get_rect()
        self.size = self.image.get_size()
        self.rect.midbottom = (midbottom_x, midbottom_y)
        self.velocity1 = 5
        self.velocity2 = 10
        self.distance_for_super_speed = random.randint(int(self.size[1]+self.size[1]/2), 2 * self.size[1])

    def move(self):
        if self.rect.bottom > self.distance_for_super_speed:
            self.rect.top += self.velocity2
        else:
            self.rect.top += self.velocity1

    def draw(self, screen_surface):
        screen_surface.blit(self.image, self.rect)

    def update(self, screen_surface, cont_frame):
        self.move()
        self.draw(screen_surface)
        if self.rect.top > utils.SCREEN_HEIGHT:
            self.kill()

    def fire(self):
        return EnemyFastBullet(self.rect)
