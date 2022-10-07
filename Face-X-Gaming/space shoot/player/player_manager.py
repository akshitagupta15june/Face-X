import pygame
from .player import Player
import utils

class PlayerManager:
    def __init__(self):
        self._player = None
        self._player_group = pygame.sprite.Group()
        self._bullet_group = pygame.sprite.Group()
        self._score = 0

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        self._score = value

    @property
    def player(self):
        return self._player

    @property
    def player_group(self):
        return self._player_group

    @property
    def bullet_group(self):
        return self._bullet_group

    def create(self, x, y):
        self.player_group.add(Player(x, y))
        self.set_player()

    def update(self, screen_surface, x, y, target):
        self.player_group.update(screen_surface, x, y, target)
        self.player_group.draw(screen_surface)

    def set_player(self):
        if len(self.player_group.sprites()) > 0:
            self._player = self.player_group.sprites()[0]

    def fire(self, screen_surface, cont_frame):
        if cont_frame % 10 == 0:
            self.bullet_group.add(self.player.fire())
        self.bullet_group.draw(screen_surface)
        self.bullet_group.update(screen_surface)

    def destroy(self, enemies, enemy_bullets):
        if pygame.sprite.groupcollide(self.player_group, enemies, True, True) or \
                pygame.sprite.groupcollide(self.player_group, enemy_bullets, True, True):
                death = pygame.mixer.Sound("assets/sounds/player_death.mp3")
                death.play()
                return True
        return False


