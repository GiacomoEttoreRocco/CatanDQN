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
    def __init__(self, game):
        # #Use pygame to display the board
        self.game = game #?????
        windowSize = 1000, 800
        self.playerColorDict = {0: pygame.Color('grey'), 1: pygame.Color('red'), 2: pygame.Color('yellow'),
                           3: pygame.Color('blueviolet'), 4:  pygame.Color('blue')}
        self.tileColorDict = {"clay": (188, 74, 60), "iron": (128, 128, 128), "crop": pygame.Color('orange'), "wood": (0, 153, 0),
                         "sheep": (51, 255, 51), "desert": (245, 222, 179) }
        self.imgDict = {"clay": "imgs/clay.png", "iron": "imgs/iron.png", "crop": "imgs/crop.png", "wood": "imgs/wood.png",
                   "sheep": "imgs/sheep.png", "desert": "imgs/desert.png"}
        self.graphicTileList = []
        self.graphicPlaceList = []
        self.screen = pygame.display.set_mode(windowSize)
        self.font_resource = pygame.font.SysFont('tahoma', 55)
        self.font_resourceSmaller = pygame.font.SysFont('tahoma', 35)
        self.font_resourceSmallest = pygame.font.SysFont('tahoma', 17, bold=True)


        self.font_harbors = pygame.font.SysFont('tahoma', 15)
        self.font_robber = pygame.font.SysFont('tahoma', 50)

        self.pointsP1 = self.font_resource.render(str(self.game.players[0].victoryPoints), False, self.playerColorDict[1])
        self.pointsP2 = self.font_resource.render(str(self.game.players[1].victoryPoints), False, self.playerColorDict[2])
        self.pointsP3 = self.font_resource.render(str(self.game.players[2].victoryPoints), False, self.playerColorDict[3])
        self.pointsP4 = self.font_resource.render(str(self.game.players[3].victoryPoints), False, self.playerColorDict[4])
        self.bgScoreColor = pygame.Color("black")


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
            pygame.draw.polygon(self.screen, pygame.Color(tileColorRGB[0], tileColorRGB[1], tileColorRGB[2]), hexTileCorners, width == 0)
            pygame.draw.polygon(self.screen, pygame.Color('black'), hexTileCorners, 5)
            graphicTile.pixelCenter = geomlib.hex_to_pixel(hexLayout, graphicTile.hex)
            tileNumberText = self.font_resourceSmallest.render(str(boardtile.number), False, pygame.Color("black"))
            sourceFileDir = os.path.dirname(os.path.abspath(__file__))
            imgPath = os.path.join(sourceFileDir, self.imgDict[boardtile.resource])
            image = pygame.image.load(imgPath).convert_alpha()
            mask = image.copy()
            mask = pygame.transform.scale(mask, (130, 130))
            self.screen.blit(mask, (graphicTile.pixelCenter.x - 65, graphicTile.pixelCenter.y - 65))
            if boardtile.resource != 'desert':
                pygame.draw.circle(self.screen, pygame.Color("black"), (graphicTile.pixelCenter.x, graphicTile.pixelCenter.y+30), 17, width==0)
                pygame.draw.circle(self.screen, pygame.Color("white"), (graphicTile.pixelCenter.x, graphicTile.pixelCenter.y+30), 13, width==0)
                if(boardtile.number >= 10):
                    self.screen.blit(tileNumberText, (graphicTile.pixelCenter.x-10, graphicTile.pixelCenter.y+18))
                else:
                    self.screen.blit(tileNumberText, (graphicTile.pixelCenter.x-5, graphicTile.pixelCenter.y+18))

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
        self.updateGameScreen()
        pygame.display.update()
        event = pygame.event.wait()
        #if event.type == pygame.QUIT:
        #    running = False
        #pygame.quit()
        return

    def updateGameScreen(self):

        score = pygame.Rect(0,0,100,100) 
        self.screen.fill(self.bgScoreColor, score)  #pygame.Color('lightblue')
        pygame.display.update(score)

        score = pygame.Rect(0,700,100,100) 
        self.screen.fill(self.bgScoreColor, score)  #pygame.Color('lightblue')
        pygame.display.update(score)

        score = pygame.Rect(900,0,100,100) 
        self.screen.fill(self.bgScoreColor, score)  #pygame.Color('lightblue')
        pygame.display.update(score)

        score = pygame.Rect(900,700,100,100) 
        self.screen.fill(self.bgScoreColor, score)  #pygame.Color('lightblue')
        pygame.display.update(score)

        self.pointsP1 = self.font_resource.render(str(self.game.players[0].victoryPoints), False, self.playerColorDict[1])
        self.pointsCardsP1 = self.font_resourceSmaller.render(str(self.game.players[0].victoryPointsCards), False, self.playerColorDict[1])
        self.screen.blit(self.pointsP1, (5, 5))
        self.screen.blit(self.pointsCardsP1, (5, 60))

        self.pointsP2 = self.font_resource.render(str(self.game.players[1].victoryPoints), False, self.playerColorDict[2])
        self.pointsCardsP2 = self.font_resourceSmaller.render(str(self.game.players[1].victoryPointsCards), False, self.playerColorDict[2])
        self.screen.blit(self.pointsP2, (5, 700))
        self.screen.blit(self.pointsCardsP2, (5, 755))


        self.pointsP3 = self.font_resource.render(str(self.game.players[2].victoryPoints), False, self.playerColorDict[3])
        self.pointsCardsP3 = self.font_resourceSmaller.render(str(self.game.players[2].victoryPointsCards), False, self.playerColorDict[3])
        self.screen.blit(self.pointsP3, (950, 5))
        self.screen.blit(self.pointsCardsP3, (950, 60))

        self.pointsP4 = self.font_resource.render(str(self.game.players[3].victoryPoints), False, self.playerColorDict[4])
        self.pointsCardsP4 = self.font_resourceSmaller.render(str(self.game.players[3].victoryPointsCards), False, self.playerColorDict[4])
        self.screen.blit(self.pointsP4, (950, 700))
        self.screen.blit(self.pointsCardsP4, (950, 755))

        self.checkAndDrawStreets()
        self.checkAndDrawPlaces()
        self.drawRobber()
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
        pygame.draw.line(self.screen, pygame.Color("Black"), self.graphicPlaceList[startPos].coords, self.graphicPlaceList[endPos].coords, 20)
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
        sourceFileDir = os.path.dirname(os.path.abspath(__file__))
        imgPath = os.path.join(sourceFileDir, "imgs/robber.png")
        robberImg = pygame.image.load(imgPath).convert_alpha()
        for graphicTile in self.graphicTileList:
            if(graphicTile.robber):
                robberCoords = graphicTile.pixelCenter
                self.screen.blit(robberImg, robberCoords)

    def getHexCoords(self, hex_i):
        coordDict = {0: geomlib.Axial_Point(0, -2), 1: geomlib.Axial_Point(1, -2), 2: geomlib.Axial_Point(2, -2),
                     3: geomlib.Axial_Point(-1, -1), 4: geomlib.Axial_Point(0, -1), 5: geomlib.Axial_Point(1, -1),
                     6: geomlib.Axial_Point(2, -1), 7: geomlib.Axial_Point(-2, 0), 8: geomlib.Axial_Point(-1, 0), 9: geomlib.Axial_Point(0, 0),
                     10: geomlib.Axial_Point(1, 0), 11: geomlib.Axial_Point(2, 0), 12: geomlib.Axial_Point(-2, 1),
                     13: geomlib.Axial_Point(-1, 1), 14: geomlib.Axial_Point(0, 1), 15: geomlib.Axial_Point(1, 1), 16: geomlib.Axial_Point(-2, 2),
                     17: geomlib.Axial_Point(-1, 2), 18: geomlib.Axial_Point(0, 2)}
        return coordDict[hex_i]