import random
import pygame
from .enemy_standard import EnemyStandard
from .enemy_fast import EnemyFast
from .enemy_explosion import EnemyExlosion
import utils


class EnemiesManager:
    def __init__(self):
        self._enemy_group = pygame.sprite.Group()
        self._bullet_group = pygame.sprite.Group()
        self.explosion_group = pygame.sprite.Group()

        self.enemy_breeding_places = [{'midbottom_x': 40, 'midbottom_y': 0},
                                      {'midbottom_x': 80, 'midbottom_y': 0},
                                      {'midbottom_x': 120, 'midbottom_y': 0},
                                      {'midbottom_x': 160, 'midbottom_y': 0},
                                      {'midbottom_x': 200, 'midbottom_y': 0},
                                      {'midbottom_x': 240, 'midbottom_y': 0},
                                      {'midbottom_x': 280, 'midbottom_y': 0},
                                      {'midbottom_x': 320, 'midbottom_y': 0},
                                      {'midbottom_x': 360, 'midbottom_y': 0}]


    @property
    def enemy_group(self):
        return self._enemy_group

    @property
    def bullet_group(self):
        return self._bullet_group

    def create(self, cont_frame):
        if cont_frame % 20 == 0:
            next_place = self.enemy_breeding_places[random.randint(0, len(self.enemy_breeding_places)-1)]
            enemy = random.randint(1, 2)
            if enemy == 1:
                self.enemy_group.add(EnemyStandard(next_place['midbottom_x'], next_place['midbottom_y']))
            if enemy == 2:
                self.enemy_group.add(EnemyFast(next_place['midbottom_x'], next_place['midbottom_y']))

    def update(self, screen_surface, cont_frame):
        self.enemy_group.update(screen_surface, cont_frame)
        self.enemy_group.draw(screen_surface)

    def fire(self, screen_surface, cont_frame):
        if cont_frame % 30 == 0:
            for enemy in self.enemy_group:
                self.bullet_group.add(enemy.fire())
        self.bullet_group.draw(screen_surface)
        self.bullet_group.update(screen_surface)

    def destroy(self, screen_surface, bullet_group):
        enemy_defeated = pygame.sprite.groupcollide(self.enemy_group, bullet_group, True, True)

        enemy = [k for k in enemy_defeated.keys()]
        if len(enemy) > 0:
            x = enemy[0].rect.centerx
            y = enemy[0].rect.centery
            self.explosion_group.add(EnemyExlosion(x, y))
        self.explosion_group.update(screen_surface)

        if enemy_defeated:
            defeated = pygame.mixer.Sound("assets/sounds/enemy_death.mp3")
            defeated.play()
            return 1
        return 0


