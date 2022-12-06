
class GraphicPlace:
    def __init__(self, place, coords):
        self.coords = coords
        self.index = place.id
        self.hasHarbor = False if place.harbor == "" else True
        self.isOnTheSea = place.isOnTheSea()