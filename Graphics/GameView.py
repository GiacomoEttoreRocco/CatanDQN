import pygame
import Graphics.GraphicTile as GraphicTile
import Graphics.GraphicPlace as GraphicPlace
import Graphics.geomlib as geomlib
import Classes.CatanGraph  as cg
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
        self.font_harbors = pygame.font.SysFont('cambria', 10)

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
        flat = geomlib.Layout(geomlib.layout_pointy, geomlib.Point(80, 80),
                      geomlib.Point(500, 400))  # specify Layout
        print(Board().tiles)
        width = 1000
        hex_i = 0
        for boardtile in Board().tiles:
            hexCoords = self.getHexCoords(hex_i)
            graphicTile = GraphicTile.GraphicTile(hexCoords, boardtile)
            self.graphicTileList.append(graphicTile)
            hexTileCorners = geomlib.polygon_corners(flat, graphicTile.hex)
            tileColorRGB = colorDict_RGB[boardtile.resource]
            pygame.draw.polygon(self.screen, pygame.Color(tileColorRGB[0], tileColorRGB[1], tileColorRGB[2]), hexTileCorners, width == 0)
            graphicTile.pixelCenter = geomlib.hex_to_pixel(flat, graphicTile.hex)
            resourceText = self.font_resource.render(str(boardtile.resource) + " \n" +str(boardtile.number), False, (0, 0, 0))
            self.screen.blit(resourceText, (graphicTile.pixelCenter.x - 25, graphicTile.pixelCenter.y))
            hex_i += 1
        return None

    def setupInitialPlaces(self):
        print(Board().places)
        for place in Board().places:
            self.graphicPlaceList.append(GraphicPlace.GraphicPlace(place))
        alreadyFound = []
        for gtile in self.graphicTileList:
            for k, v in cg.tilePlaces.items():
                if gtile.index == k:
                    for el in v:
                        if el not in alreadyFound:
                            placeToAdd = self.graphicPlaceList[el]
                            placeToAdd.setupCoords(gtile.pixelCenter.x, gtile.pixelCenter.y)        #A mano?
                            gtile.places.append(placeToAdd)
                            alreadyFound.append(el)
            print(gtile.places)

        for gtile in self.graphicTileList:
            for place in gtile.places:
                print(place.isCity)
                self.drawPlace(place)

    def displayGameScreen(self):
        # First display all initial hexes and regular buttons
        running = True
        self.displayInitialBoard()
        self.setupInitialPlaces()
        while running:
            # other code
            pygame.display.update()
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                running = False  # Be interpreter friendly
        pygame.quit()

        return

    def getHexCoords(self, hex_i):
        coordDict = {0: geomlib.Axial_Point(-1, -1), 1: geomlib.Axial_Point(-2, 0), 2: geomlib.Axial_Point(-2, 1), 3: geomlib.Axial_Point(-2, 2), 4: geomlib.Axial_Point(-1, 2), 5: geomlib.Axial_Point(0, 2), 6: geomlib.Axial_Point(1, 1), 7: geomlib.Axial_Point(2, 0), 8: geomlib.Axial_Point(2, -1), 9: geomlib.Axial_Point(2,-2), 10: geomlib.Axial_Point(1, -2),
                        11: geomlib.Axial_Point(0, -2), 12: geomlib.Axial_Point(-1, 0), 13: geomlib.Axial_Point(-1, 1), 14: geomlib.Axial_Point(0, 1), 15: geomlib.Axial_Point(1, 0), 16: geomlib.Axial_Point(1, -1), 17: geomlib.Axial_Point(0, -1), 18: geomlib.Axial_Point(0, 0)}
        return coordDict[hex_i]

    def drawPlace(self, graphicPlace):
        print(graphicPlace.harbor)
        if graphicPlace.harbor is not None:
            harborText = self.font_harbors.render(graphicPlace.harbor, False, (0, 0, 0))
            self.screen.blit(harborText, graphicPlace.coords)
        color = pygame.Color('grey')    #Needs to be taken from player
        if graphicPlace.isColony:
            pygame.draw.rect(self.screen, color, (graphicPlace.coords[0] - 5, graphicPlace.coords[1], 20, 20))
        elif graphicPlace.isCity:
            pygame.draw.circle(self.screen, color, graphicPlace.coords, 10)
        pygame.draw.circle(self.screen, color, graphicPlace.coords, 10)     #da togliere!

    def placeRobber(self):
        robberText = self.font_Robber.render("R", False, (0, 0, 0))
        for graphicTile in self.graphicTileList:
            if(graphicTile.robber):
                robberCoords = graphicTile.pixelCenter
                self.screen.blit(robberText, (int(robberCoords.x) -20, int(robberCoords.y)-35))