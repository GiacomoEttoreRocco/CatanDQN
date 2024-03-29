import Graphics.geomlib as geomlib

class GraphicTile():
    def __init__(self, axialCoords, tile):
        self.hex = geomlib.Axial_Hex(axialCoords)  # Hex representation of this tile
        self.resource = tile.resource
        self.coord = axialCoords
        self.pixelCenter = None  # Pixel coordinates of hex as Point(x, y)
        self.index = tile.identificator
        self.adjacentTiles = []
        self.places = []
        self.robber = False
        self.number = tile.number

    # Function to Display Hex Info
    def displayHexInfo(self):
        print('Index:{}; Hex:{}; Axial Coord:{}'.format(self.index, self.resource, self.coord))
        return None