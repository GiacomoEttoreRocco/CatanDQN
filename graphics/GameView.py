import pygame
from GraphicTile import *
from GraphicPlace import *
from lib2 import *
#from Classes.CatanGraph import CatanGraph
from Classes.Board import Board

pygame.init()

class GameView:
    def __init__(self):
        # #Use pygame to display the board
        size = 1000, 800
        self.graphicTileList = []
        self.graphicPlaceList = []
        self.screen = pygame.display.set_mode(size)
        self.font_resource = pygame.font.SysFont('cambria', 15)
        self.font_ports = pygame.font.SysFont('cambria', 10)

        self.font_button = pygame.font.SysFont('cambria', 12)
        self.font_diceRoll = pygame.font.SysFont('cambria', 25)
        self.font_Robber = pygame.font.SysFont('arialblack', 50)

    # Function to display the initial board
    def displayInitialBoard(self):
        # Dictionary to store RGB Color values
        colorDict_RGB = {"clay": (255, 51, 51), "iron": (128, 128, 128), "crop": (255, 255, 51), "wood": (0, 153, 0),
                         "sheep": (51, 255, 51), "desert": (255, 255, 204)}
        pygame.draw.rect(self.screen, pygame.Color('white'),
                         (0, 0, 1000, 800))  # blue background

        # Render each hexTile
        flat = Layout(layout_pointy, Point(80, 80),
                      Point(500, 400))  # specify Layout
        print(Board().tiles)
        width = 1000
        hex_i = 0
        for boardtile in Board().tiles:
            hexCoords = self.getHexCoords(hex_i)
            graphicTile = GraphicTile(hexCoords, boardtile)
            self.graphicTileList.append(graphicTile)
            hexTileCorners = polygon_corners(flat, graphicTile.hex)
            tileColorRGB = colorDict_RGB[boardtile.resource]
            pygame.draw.polygon(self.screen, pygame.Color(tileColorRGB[0], tileColorRGB[1], tileColorRGB[2]), hexTileCorners, width == 0)
            graphicTile.pixelCenter = hex_to_pixel(flat, graphicTile.hex)
            resourceText = self.font_resource.render(str(boardtile.resource) + " \n" +str(boardtile.number), False, (0, 0, 0))
            self.screen.blit(resourceText, (graphicTile.pixelCenter.x - 25, graphicTile.pixelCenter.y))
            hex_i += 1
        return None

    def setupInitialPlaces(self):
        for gtile in self.graphicTileList:
            gtile.places.append(GraphicPlace(Board().places[0], ((gtile.pixelCenter.x - 70), (gtile.pixelCenter.y - 40))))
            gtile.places.append(GraphicPlace(Board().places[0], ((gtile.pixelCenter.x + 70), (gtile.pixelCenter.y + 40))))
            gtile.places.append(GraphicPlace(Board().places[0], ((gtile.pixelCenter.x - 70), (gtile.pixelCenter.y + 40))))
            gtile.places.append(GraphicPlace(Board().places[0], ((gtile.pixelCenter.x + 70), (gtile.pixelCenter.y - 40))))
            gtile.places.append(GraphicPlace(Board().places[0], (gtile.pixelCenter.x, (gtile.pixelCenter.y + 75))))
            gtile.places.append(GraphicPlace(Board().places[0], (gtile.pixelCenter.x, (gtile.pixelCenter.y - 75))))


    def displayGameScreen(self):
        # First display all initial hexes and regular buttons
        self.displayInitialBoard()
        self.setupInitialPlaces()
        pygame.display.update()
        return

    def getHexCoords(self, hex_i):
        coordDict = {0: Axial_Point(-1, -1), 1: Axial_Point(-2, 0), 2: Axial_Point(-2, 1), 3: Axial_Point(-2, 2), 4: Axial_Point(-1, 2), 5: Axial_Point(0, 2), 6: Axial_Point(1, 1), 7: Axial_Point(2, 0), 8: Axial_Point(2, -1), 9: Axial_Point(2,-2), 10: Axial_Point(1, -2),
                        11: Axial_Point(0, -2), 12: Axial_Point(-1, 0), 13: Axial_Point(-1, 1), 14: Axial_Point(0, 1), 15: Axial_Point(1, 0), 16:Axial_Point(1, -1), 17:Axial_Point(0, -1), 18: Axial_Point(0, 0)}
        return coordDict[hex_i]


    def drawPlace(self, place):
        pygame.draw.circle(self.screen, pygame.Color('grey'), place.coords, 10)

    def placeRobber(self):
        robberText = self.font_Robber.render("R", False, (0, 0, 0))
        for graphicTile in self.graphicTileList:
            if(graphicTile.robber):
                robberCoords = graphicTile.pixelCenter
                self.screen.blit(robberText, (int(robberCoords.x) -20, int(robberCoords.y)-35))