import pygame
import os

class GraphicPlace:
    def __init__(self, place):
        self.coords = None
        self.index = place.id
        self.isColony = place.isColony
        self.isCity = place.isCity
        self.owner = place.owner
        self.harbor = None if place.harbor == "" else place.harbor
        self.sprite = None
        self.isOnTheSea = place.isOnTheSea()

    def setupCoords(self, coords):
        self.coords = (coords[0], coords[1])

    def update(self, place):
        self.isColony = place.isColony
        self.isCity = place.isCity
        self.owner = place.owner

    def setupSprite(self, color):
        sourceFileDir = os.path.dirname(os.path.abspath(__file__))
        imgPath = os.path.join(sourceFileDir, "imgs/casa.png")     #"cityImg.png" if self.isCity else "colonyImg.png"
        self.sprite = PlaceSprite(imgPath, pygame.Color('red'), self.coords)    #Player's color


class PlaceSprite(pygame.sprite.Sprite):

    def __init__(self, imgName, color, coords):
        super().__init__()

        self.image = pygame.image.load(imgName)
        self.image = pygame.transform.scale(self.image, (50, 50))
        self.rect = self.image.get_rect()
        self.rect.centerx = coords[0]
        self.rect.centery = coords[1]
        colorFill = pygame.Surface(self.image.get_size()).convert_alpha()
        self.image.fill(color)
        self.image.blit(colorFill, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)



