import pygame
import utils

class PlayerBullet(pygame.sprite.Sprite):
    def __init__(self, player_rect, posx):
        super(PlayerBullet, self).__init__()
        self.image = pygame.image.load("assets/sprites/space-bullets/player-bullet-standard.png").convert_alpha()
        self.rect = self.image.get_rect()
        self.rect.centerx = player_rect.centerx + posx
        self.rect.centery = player_rect.centery
        self.speed = 15

    def update(self, screen_surface):
        self.move()
        if self.rect.top < 0:
            self.kill()

    def move(self):
        self.rect.centery -= self.speed

    def draw(self, screen_surface):
        screen_surface.blit(self.image, self.rect)
