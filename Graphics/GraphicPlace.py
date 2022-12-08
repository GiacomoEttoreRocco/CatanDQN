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

    def setupSprite(self):
        sourceFileDir = os.path.dirname(os.path.abspath(__file__))
        if self.isColony:
            imgPath = os.path.join(sourceFileDir, "imgs\playericons\set_p" + str(self.owner) + ".png")
        elif self.isCity:
            imgPath = os.path.join(sourceFileDir, "imgs\playericons\cit_p" + str(self.owner) + ".png")
        self.sprite = PlaceSprite(imgPath, self.coords)


class PlaceSprite(pygame.sprite.Sprite):

    def __init__(self, imgName, coords):
        super().__init__()
        self.image = pygame.image.load(imgName)
        self.image = pygame.transform.scale(self.image, (50, 50))
        self.rect = self.image.get_rect()
        self.rect.centerx = coords[0]
        self.rect.centery = coords[1]



