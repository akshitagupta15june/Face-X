import pygame
import utils


class Target(pygame.sprite.Group):
    def __init__(self):
        super(Target, self).__init__()
        self.target_surfaces = [pygame.image.load("assets/sprites/targets/target-circle.png"),
                                pygame.image.load("assets/sprites/targets/target-tri.png")]

        self.targets = []
        for surface in self.target_surfaces:
            self.targets.append([surface, surface.get_rect()])

        self._target_on_player = False
        self.atual_target = self.targets[0]

    @property
    def target_on_player(self):
        return self._target_on_player

    @target_on_player.setter
    def target_on_player(self, value):
        self._target_on_player = value

    def set_target_on_player(self, player_rect):
        if (player_rect.centerx + 30 >= self.atual_target[1].centerx >= player_rect.centerx - 30) and \
           (player_rect.centery + 30 >= self.atual_target[1].centery >= player_rect.centery - 30):
            self.target_on_player = True
        else:
            self.target_on_player = False

    def set_atual_target(self):
        if self.target_on_player:
            self.atual_target = self.targets[1]
        else:
            self.atual_target = self.targets[0]

    def draw(self, screen_surface):
        screen_surface.blit(self.atual_target[0], self.atual_target[1])

    def move(self, x, y):
        self.atual_target[1].centerx = x
        self.atual_target[1].centery = y

    def update(self, screen_surface, x, y, player_rect):
        self.set_target_on_player(player_rect)
        self.set_atual_target()
        self.move(x, y)
        self.draw(screen_surface)


