import pygame
import Graphics.GraphicTile as GraphicTile
import Graphics.GraphicPlace as GraphicPlace
import Graphics.geomlib as geomlib
import Classes.CatanGraph  as cg
import Classes.Board as Board
import Graphics.PlaceCoordinates as pc
import os
import time

pygame.init()

class GameView:
    def __init__(self):
        # #Use pygame to display the board
        windowSize = 1000, 800
        self.playerColorDict = {0: pygame.Color('grey'), 1: pygame.Color('red'), 2: pygame.Color('yellow'),
                           3: pygame.Color('green'), 4:  pygame.Color('blue')}
        self.tileColorDict = {"clay": (188, 74, 60), "iron": (128, 128, 128), "crop": pygame.Color('orange'), "wood": (0, 153, 0),
                         "sheep": (51, 255, 51), "desert": (245, 222, 179) }
        self.imgDict = {"clay": "imgs/clay.png", "iron": "imgs/iron.png", "crop": "imgs/crop.png", "wood": "imgs/wood.png",
                   "sheep": "imgs/sheep.png", "desert": "imgs/desert.png"}
        self.graphicTileList = []
        self.graphicPlaceList = []
        self.screen = pygame.display.set_mode(windowSize)
        self.font_resource = pygame.font.SysFont('tahoma', 55)
        self.font_harbors = pygame.font.SysFont('tahoma', 15)
        self.font_robber = pygame.font.SysFont('tahoma', 50)

    def setupAndDisplayBoard(self):
        #Draw the sea
        pygame.draw.rect(self.screen, pygame.Color('lightblue'),
                         (0, 0, 1000, 800))

        # Render each tile
        hexLayout = geomlib.Layout(geomlib.layout_pointy, geomlib.Point(80, 80),
                      geomlib.Point(500, 400))
        width = 1000
        hex_i = 0
        #Takes tiles from board and draws their graphic equivalent
        for boardtile in Board.Board().tiles:
            hexCoords = self.getHexCoords(hex_i)
            graphicTile = GraphicTile.GraphicTile(hexCoords, boardtile)
            self.graphicTileList.append(graphicTile)
            hexTileCorners = geomlib.polygon_corners(hexLayout, graphicTile.hex)
            tileColorRGB = self.tileColorDict[boardtile.resource]
            #Draw hexagonal tile
            pygame.draw.polygon(self.screen, pygame.Color(tileColorRGB[0], tileColorRGB[1], tileColorRGB[2]), hexTileCorners, width == 0)
            pygame.draw.polygon(self.screen, pygame.Color('black'), hexTileCorners, 5)
            #Position the tile
            graphicTile.pixelCenter = geomlib.hex_to_pixel(hexLayout, graphicTile.hex)
            resourceText = self.font_resource.render(str(boardtile.number), False, (255, 255, 255))
            #Setup images
            sourceFileDir = os.path.dirname(os.path.abspath(__file__))
            imgPath = os.path.join(sourceFileDir, self.imgDict[boardtile.resource])
            image = pygame.image.load(imgPath).convert_alpha()
            mask = image.copy()
            mask = pygame.transform.scale(mask, (130, 130))
            self.screen.blit(mask, (graphicTile.pixelCenter.x - 65, graphicTile.pixelCenter.y - 65))
            if boardtile.resource != 'desert':
                self.screen.blit(resourceText, (graphicTile.pixelCenter.x - 25, graphicTile.pixelCenter.y - 30))
            hex_i += 1
        return None

    def setupPlaces(self):
        for place in Board.Board().places:
            self.graphicPlaceList.append(GraphicPlace.GraphicPlace(place))
        #Each tile has 6 places, so here it reads which places belong to a tile and then assigns them and gives them coordinates
        alreadyFound = []
        for gtile in self.graphicTileList:
            for k, v in cg.tilePlaces.items():
                if gtile.index == k:
                    for el in v:
                        if el not in alreadyFound:
                            placeToAdd = self.graphicPlaceList[el]
                            placeToAdd.setupCoords(pc.placeCoordinates[placeToAdd.index])        
                            gtile.places.append(placeToAdd)
                            alreadyFound.append(el)

    def displayGameScreen(self):
        running = True
        self.setupAndDisplayBoard()
        self.setupPlaces()
        #Check turn end
        self.updateGameScreen()
        #while running:
        pygame.display.update()
        event = pygame.event.wait()
        #if event.type == pygame.QUIT:
        #    running = False
        #pygame.quit()
        return

    def updateGameScreen(self):
        self.checkAndDrawPlaces()
        self.checkAndDrawStreets()
        pygame.display.update()
        time.sleep(0.1)

    def drawPlace(self, graphicPlace):
        if graphicPlace.harbor is not None:
            harborText = self.font_harbors.render(graphicPlace.harbor, False, (0, 0, 0))
            self.screen.blit(harborText, (graphicPlace.coords[0] + 10, graphicPlace.coords[1] + 10))
        graphicPlace.setupSprite()
        sprlist = pygame.sprite.Group()
        sprlist.add(graphicPlace.sprite)
        sprlist.draw(self.screen)

    def drawStreet(self, edge, color):
        startPos = edge[0]
        endPos = edge[1]
        pygame.draw.line(self.screen, color, self.graphicPlaceList[startPos].coords, self.graphicPlaceList[endPos].coords, 10)

    def checkAndDrawPlaces(self):
        for gplace, place in zip(self.graphicPlaceList, Board.Board().places):
            gplace.update(place)
            if place.owner != 0:
                self.drawPlace(gplace) #    self.drawPlace(place)
                #print(gplace.isColony)
                #print(gplace.isCity)

    def checkAndDrawStreets(self):
        for edge in Board.Board().edges:
            owner = Board.Board().edges[edge]
            if owner != 0:
                self.drawStreet(edge, self.playerColorDict[owner])
                #print("PRINT OWNER: ", owner)

    def drawRobber(self):
        robberText = self.font_robber.render("Robber", False, (0, 0, 0))
        for graphicTile in self.graphicTileList:
            if(graphicTile.robber):
                robberCoords = graphicTile.pixelCenter
                self.screen.blit(robberText, (int(robberCoords.x) - 20, int(robberCoords.y) - 35))

    def getHexCoords(self, hex_i):
        coordDict = {0: geomlib.Axial_Point(0, -2), 1: geomlib.Axial_Point(1, -2), 2: geomlib.Axial_Point(2, -2),
                     3: geomlib.Axial_Point(-1, -1), 4: geomlib.Axial_Point(0, -1), 5: geomlib.Axial_Point(1, -1),
                     6: geomlib.Axial_Point(2, -1), 7: geomlib.Axial_Point(-2, 0), 8: geomlib.Axial_Point(-1, 0), 9: geomlib.Axial_Point(0, 0),
                     10: geomlib.Axial_Point(1, 0), 11: geomlib.Axial_Point(2, 0), 12: geomlib.Axial_Point(-2, 1),
                     13: geomlib.Axial_Point(-1, 1), 14: geomlib.Axial_Point(0, 1), 15: geomlib.Axial_Point(1, 1), 16: geomlib.Axial_Point(-2, 2),
                     17: geomlib.Axial_Point(-1, 2), 18: geomlib.Axial_Point(0, 2)}
        return coordDict[hex_i]