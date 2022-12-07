
class GraphicPlace:
    def __init__(self, place):
        self.coords = None
        self.index = place.id
        self.isColony = place.isColony
        self.isCity = place.isCity
        self.owner = place.owner
        self.harbor = None if place.harbor == "" else place.harbor
        self.isOnTheSea = place.isOnTheSea()

    def setupCoords(self, coords):
        self.coords = (coords[0], coords[1])