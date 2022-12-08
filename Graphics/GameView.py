import pygame
import Graphics.GraphicTile as GraphicTile
import Graphics.GraphicPlace as GraphicPlace
import Graphics.geomlib as geomlib
import Classes.CatanGraph  as cg
import Classes.Board as Board
import Graphics.PlaceCoordinates as pc
import os

pygame.init()

class GameView:
    def __init__(self):
        # #Use pygame to display the board
        size = 1000, 800
        self.playerColorDict = {0: pygame.Color('grey'), 1: pygame.Color('red'), 2: pygame.Color('royalblue'),
                           3: pygame.Color('green'), 4:  pygame.Color('yellow')}
        self.edges = Board.Board().edges
        self.graphicTileList = []
        self.graphicPlaceList = []
        self.screen = pygame.display.set_mode(size)
        self.font_resource = pygame.font.SysFont('tahoma', 55)
        self.font_harbors = pygame.font.SysFont('tahoma', 15)
        self.font_robber = pygame.font.SysFont('tahoma', 50)

    # Function to display the initial board
    def displayInitialBoard(self):
        # Dictionary to store RGB Color values
        colorDict_RGB = {"clay": (188, 74, 60), "iron": (128, 128, 128), "crop": pygame.Color('orange'), "wood": (0, 153, 0),
                         "sheep": (51, 255, 51), "desert": (245, 222, 179) }
        imgDict = {"clay": "imgs/clay.png", "iron": "imgs/iron.png", "crop": "imgs/crop.png", "wood": "imgs/wood.png",
                   "sheep": "imgs/sheep.png", "desert": "imgs/desert.png"}
        pygame.draw.rect(self.screen, pygame.Color('lightblue'),
                         (0, 0, 1000, 800))  # blue background

        # Render each hexTile
        flat = geomlib.Layout(geomlib.layout_pointy, geomlib.Point(80, 80),
                      geomlib.Point(500, 400))  # specify Layout
        #print(Board.Board().tiles)
        width = 1000
        hex_i = 0
        for boardtile in Board.Board().tiles:
            hexCoords = self.getHexCoords(hex_i)
            graphicTile = GraphicTile.GraphicTile(hexCoords, boardtile)
            self.graphicTileList.append(graphicTile)
            hexTileCorners = geomlib.polygon_corners(flat, graphicTile.hex)
            tileColorRGB = colorDict_RGB[boardtile.resource]
            pygame.draw.polygon(self.screen, pygame.Color(tileColorRGB[0], tileColorRGB[1], tileColorRGB[2]), hexTileCorners, width == 0)
            pygame.draw.polygon(self.screen, pygame.Color('black'), hexTileCorners, 5)
            graphicTile.pixelCenter = geomlib.hex_to_pixel(flat, graphicTile.hex)
            resourceText = self.font_resource.render(str(boardtile.number), False, (255, 255, 255))
            sourceFileDir = os.path.dirname(os.path.abspath(__file__))
            imgPath = os.path.join(sourceFileDir, imgDict[boardtile.resource])
            image = pygame.image.load(imgPath).convert_alpha()
            mask = image.copy()
            mask = pygame.transform.scale(mask, (130, 130))
            self.screen.blit(mask, (graphicTile.pixelCenter.x - 65, graphicTile.pixelCenter.y - 65))
            if boardtile.resource != 'desert':
                self.screen.blit(resourceText, (graphicTile.pixelCenter.x - 25, graphicTile.pixelCenter.y - 30))
            hex_i += 1


        return None

    def setupInitialPlaces(self):
        #print(Board.Board().places)
        for place in Board.Board().places:
            self.graphicPlaceList.append(GraphicPlace.GraphicPlace(place))
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

        color = pygame.Color('grey')
        for gtile in self.graphicTileList:
            for place in gtile.places:
                self.drawPlace(place, color)

    def displayGameScreen(self):
        # First display all initial hexes and regular buttons
        running = True
        self.displayInitialBoard()
        self.setupInitialPlaces()
        #Check turn end
        self.updateScreen()
        while running:
            # other code
            pygame.display.update()
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                running = False  # Be interpreter friendly
        pygame.quit()

        return

    def getHexCoords(self, hex_i):
        coordDict = {0: geomlib.Axial_Point(0, -2), 1: geomlib.Axial_Point(1, -2), 2: geomlib.Axial_Point(2, -2),
                     3: geomlib.Axial_Point(-1, -1), 4: geomlib.Axial_Point(0, -1), 5: geomlib.Axial_Point(1, -1),
                     6: geomlib.Axial_Point(2, -1), 7: geomlib.Axial_Point(-2, 0), 8: geomlib.Axial_Point(-1, 0), 9: geomlib.Axial_Point(0, 0),
                     10: geomlib.Axial_Point(1, 0), 11: geomlib.Axial_Point(2, 0), 12: geomlib.Axial_Point(-2, 1),
                     13: geomlib.Axial_Point(-1, 1), 14: geomlib.Axial_Point(0, 1), 15: geomlib.Axial_Point(1, 1), 16: geomlib.Axial_Point(-2, 2),
                     17: geomlib.Axial_Point(-1, 2), 18: geomlib.Axial_Point(0, 2)}
        return coordDict[hex_i]

    def updateScreen(self):
        #after the end of every turn
        for gplace, place in zip(self.graphicPlaceList, Board.Board().places):
            gplace.update(place)



    def drawPlace(self, graphicPlace, color):

        if graphicPlace.harbor is not None:
            harborText = self.font_harbors.render(graphicPlace.harbor, False, (0, 0, 0))
            self.screen.blit(harborText, (graphicPlace.coords[0] + 10, graphicPlace.coords[1] + 10))

        graphicPlace.setupSprite(self.playerColorDict[graphicPlace.owner])
        sprlist = pygame.sprite.Group()
        sprlist.add(graphicPlace.sprite)
        sprlist.draw(self.screen)
        # if graphicPlace.isColony:
        #     pygame.draw.rect(self.screen, color, (graphicPlace.coords[0] - 5, graphicPlace.coords[1], 20, 20))
        # elif graphicPlace.isCity:
        #     pygame.draw.circle(self.screen, color, graphicPlace.coords, 10)
        #Places show up only when a player plays something

    def drawStreet(self, edge, color):
        startPos = edge[0]
        endPos = edge[1]
        pygame.draw.line(self.screen, color, startPos, endPos, 8)

    def checkAndDrawStreet(self):
        for edge in self.edges:
            owner = self.edges[edge]
            self.drawStreet(edge, self.playerColorDict[owner])



    def placeRobber(self):
        robberText = self.font_robber.render("R", False, (0, 0, 0))
        for graphicTile in self.graphicTileList:
            if(graphicTile.robber):
                robberCoords = graphicTile.pixelCenter
                self.screen.blit(robberText, (int(robberCoords.x) -20, int(robberCoords.y)-35))