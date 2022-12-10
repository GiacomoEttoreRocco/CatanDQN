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
        self.sourceFileDir = os.path.dirname(os.path.abspath(__file__))
        self.robberImgPath = os.path.join(self.sourceFileDir, "imgs/robber.png")
        self.tempRobberTile = -1 # per motivi di efficienza.
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

        self.font_harbors = pygame.font.SysFont('tahoma', 20)
        self.font_robber = pygame.font.SysFont('tahoma', 50)

        self.points = []
        self.pointsCards = []

        self.woods = []
        self.sheeps = []
        self.crops = []
        self.irons = []
        self.clays = []
        self.knights = []

        self.turns = []
        self.bgScoreColor = pygame.Color("grey18")
        self.bgScoreColorHighlited = pygame.Color('grey34')

        for i in range(0, len(self.game.players)):
            self.points.append(self.font_resource.render(str(self.game.players[i].victoryPoints), False, self.playerColorDict[i+1]))
            self.pointsCards.append(self.font_resourceSmallest.render("Vp cards: " + str(self.game.players[i].victoryPointsCards), False, self.playerColorDict[i+1]))
            self.woods.append(self.font_resourceSmallest.render("Wood: " + str(self.game.players[i].resources["wood"]), False, self.playerColorDict[i+1]))
            self.sheeps.append(self.font_resourceSmallest.render("Sheep: " +str(self.game.players[i].resources["sheep"]), False, self.playerColorDict[i+1]))
            self.crops.append(self.font_resourceSmallest.render("Crop: " + str(self.game.players[i].resources["crop"]), False, self.playerColorDict[i+1]))
            self.irons.append(self.font_resourceSmallest.render("Iron: " + str(self.game.players[i].resources["iron"]), False, self.playerColorDict[i+1]))
            self.clays.append(self.font_resourceSmallest.render("Clay: " + str(self.game.players[i].resources["clay"]), False, self.playerColorDict[i+1]))
            self.knights.append(self.font_resourceSmallest.render("Knights: " + str(self.game.players[i].usedKnights), False, self.playerColorDict[i+1]))

    def updateStats(self):
        for i in range(0, len(self.game.players)):
            self.points[i] = self.font_resource.render(str(self.game.players[i].victoryPoints), False, self.playerColorDict[i+1])
            self.pointsCards[i] = self.font_resourceSmallest.render("Vp cards: " + str(self.game.players[i].victoryPointsCards), False, self.playerColorDict[i+1])
            self.woods[i] = self.font_resourceSmallest.render("Wood: " + str(self.game.players[i].resources["wood"]), False, self.playerColorDict[i+1])
            self.sheeps[i] = self.font_resourceSmallest.render("Sheep: " +str(self.game.players[i].resources["sheep"]), False, self.playerColorDict[i+1])
            self.crops[i] = self.font_resourceSmallest.render("Crop: " + str(self.game.players[i].resources["crop"]), False, self.playerColorDict[i+1])
            self.irons[i] = self.font_resourceSmallest.render("Iron: " + str(self.game.players[i].resources["iron"]), False, self.playerColorDict[i+1])
            self.clays[i] = self.font_resourceSmallest.render("Clay: " + str(self.game.players[i].resources["clay"]), False, self.playerColorDict[i+1])
            self.knights[i] = (self.font_resourceSmallest.render("Knights: " + str(self.game.players[i].usedKnights), False, self.playerColorDict[i+1]))


    def blit(self, player, x, y):
        playerBox = pygame.Rect(x-5, y-5, 120, 250)
        if self.game.currentTurn == player:
            self.screen.fill(self.bgScoreColorHighlited, playerBox)
        else:
            self.screen.fill(self.bgScoreColor, playerBox)
        self.screen.blit(self.points[player.id-1], (x, y)) # 5,5
        self.screen.blit(self.pointsCards[player.id-1], (x, y+55))
        self.screen.blit(self.woods[player.id-1], (x, y+95))
        self.screen.blit(self.sheeps[player.id-1], (x, y+115))
        self.screen.blit(self.crops[player.id-1], (x, y+135))
        self.screen.blit(self.irons[player.id-1], (x, y+155))
        self.screen.blit(self.clays[player.id-1], (x, y+175))
        self.screen.blit(self.knights[player.id-1], (x, y+215))


    def setupAndDisplayBoard(self):
        pygame.draw.rect(self.screen, pygame.Color('cadetblue1'),(0, 0, 1000, 800))
        hexLayout = geomlib.Layout(geomlib.layout_pointy, geomlib.Point(80, 80), geomlib.Point(500, 400))
        width = 1000
        hex_i = 0
        for boardtile in Board.Board().tiles:
            hexCoords = self.getHexCoords(hex_i)
            graphicTile = GraphicTile.GraphicTile(hexCoords, boardtile)
            self.graphicTileList.append(graphicTile)
            hexTileCorners = geomlib.polygon_corners(hexLayout, graphicTile.hex)
            tileColorRGB = self.tileColorDict[boardtile.resource]
            pygame.draw.polygon(self.screen, pygame.Color(tileColorRGB[0], tileColorRGB[1], tileColorRGB[2]), hexTileCorners, width == 0)
            pygame.draw.polygon(self.screen, pygame.Color('black'), hexTileCorners, 5)
            graphicTile.pixelCenter = geomlib.hex_to_pixel(hexLayout, graphicTile.hex)
            tileNumberText = self.font_resourceSmaller.render(str(boardtile.number), False, pygame.Color("black"))
            imgPath = os.path.join(self.sourceFileDir, self.imgDict[boardtile.resource])
            image = pygame.image.load(imgPath).convert_alpha()
            mask = image.copy()
            mask = pygame.transform.scale(mask, (54, 54))
            self.screen.blit(mask, (graphicTile.pixelCenter.x - 27, graphicTile.pixelCenter.y - 60))
            if boardtile.resource != 'desert':
                pygame.draw.circle(self.screen, pygame.Color("black"), (graphicTile.pixelCenter.x, graphicTile.pixelCenter.y+30), 27, width==0)
                pygame.draw.circle(self.screen, pygame.Color("white"), (graphicTile.pixelCenter.x, graphicTile.pixelCenter.y+30), 23, width==0)
                if(boardtile.number >= 10):
                    self.screen.blit(tileNumberText, (graphicTile.pixelCenter.x-20, graphicTile.pixelCenter.y+10))
                else:
                    self.screen.blit(tileNumberText, (graphicTile.pixelCenter.x-10, graphicTile.pixelCenter.y+10))
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

                            self.checkAndDrawHarbors(placeToAdd)
                            alreadyFound.append(el)

    def displayGameScreen(self):
        running = True
        self.setupAndDisplayBoard()
        self.setupPlaces()
        self.updateGameScreen()
        pygame.display.update()
        event = pygame.event.wait()
        return

    def updateGameScreen(self):
        self.drawRobber()
        self.checkAndDrawStreets()
        self.checkAndDrawPlaces()
        self.updateStats()
        self.blit(self.game.players[0], 5, 5)
        self.blit(self.game.players[1], 5, 555)
        self.blit(self.game.players[2], 885, 5)
        self.blit(self.game.players[3], 885, 555)

        font_dice = self.font_resourceSmaller.render(str(self.game.dice), False, pygame.Color('white'))
        diceRoll = pygame.Rect(475, 0, 50, 50)
        self.screen.fill(self.bgScoreColor, diceRoll)
        pygame.display.update(diceRoll)
        self.screen.blit(font_dice, (480, 5))

        pygame.display.update()

    def drawPlace(self, graphicPlace):
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
                self.drawPlace(gplace) 

    def checkAndDrawStreets(self):
        for edge in Board.Board().edges:
            owner = Board.Board().edges[edge]
            if owner != 0:
                self.drawStreet(edge, self.playerColorDict[owner])

    def checkAndDrawHarbors(self, place):
        if place.harbor is not None:
            harborText = self.font_harbors.render(place.harbor, False, (0, 0, 0))
            self.screen.blit(harborText, (place.coords[0] + 15, place.coords[1] + 10))

    def drawRobber(self):
        robberImg = pygame.image.load(self.robberImgPath).convert_alpha()
        if(self.tempRobberTile != Board.Board().robberTile):
            print("drowing robber...")
            robTile = Board.Board().robberTile
            for graphicTile in self.graphicTileList:
                if(graphicTile.index == robTile):
                    robberCoords = (graphicTile.pixelCenter.x, graphicTile.pixelCenter.y-30)
                    self.screen.blit(robberImg, robberCoords)
                elif(self.tempRobberTile != -1 and self.tempRobberTile == graphicTile.index):
                    width = 1000
                    ###############
                    hexLayout = geomlib.Layout(geomlib.layout_pointy, geomlib.Point(80, 80), geomlib.Point(500, 400))
                    hexTileCorners = geomlib.polygon_corners(hexLayout, graphicTile.hex)
                    tileColorRGB = self.tileColorDict[graphicTile.resource]
                    pygame.draw.polygon(self.screen, pygame.Color(tileColorRGB[0], tileColorRGB[1], tileColorRGB[2]),
                                        hexTileCorners, width == 0)
                    pygame.draw.polygon(self.screen, pygame.Color('black'), hexTileCorners, 5)
                    graphicTile.pixelCenter = geomlib.hex_to_pixel(hexLayout, graphicTile.hex)
                    ###############
                    imgPath = os.path.join(self.sourceFileDir, self.imgDict[graphicTile.resource])
                    image = pygame.image.load(imgPath).convert_alpha()
                    mask = image.copy()
                    mask = pygame.transform.scale(mask, (54, 54))
                    self.screen.blit(mask, (graphicTile.pixelCenter.x - 27, graphicTile.pixelCenter.y - 60))
                    if graphicTile.resource != 'desert':
                        width = 1000
                        pygame.draw.circle(self.screen, pygame.Color("black"), (graphicTile.pixelCenter.x, graphicTile.pixelCenter.y+30), 27, width==0)
                        pygame.draw.circle(self.screen, pygame.Color("white"), (graphicTile.pixelCenter.x, graphicTile.pixelCenter.y+30), 23, width==0)
                        tileNumberText = self.font_resourceSmaller.render(str(graphicTile.number), False, pygame.Color("black"))
                        if(graphicTile.number >= 10):
                            self.screen.blit(tileNumberText, (graphicTile.pixelCenter.x-20, graphicTile.pixelCenter.y+10))
                        else:
                            self.screen.blit(tileNumberText, (graphicTile.pixelCenter.x-10, graphicTile.pixelCenter.y+10))
            self.tempRobberTile = robTile
                

    def getHexCoords(self, hex_i):
        coordDict = {0: geomlib.Axial_Point(0, -2), 1: geomlib.Axial_Point(1, -2), 2: geomlib.Axial_Point(2, -2),
                     3: geomlib.Axial_Point(-1, -1), 4: geomlib.Axial_Point(0, -1), 5: geomlib.Axial_Point(1, -1),
                     6: geomlib.Axial_Point(2, -1), 7: geomlib.Axial_Point(-2, 0), 8: geomlib.Axial_Point(-1, 0), 9: geomlib.Axial_Point(0, 0),
                     10: geomlib.Axial_Point(1, 0), 11: geomlib.Axial_Point(2, 0), 12: geomlib.Axial_Point(-2, 1),
                     13: geomlib.Axial_Point(-1, 1), 14: geomlib.Axial_Point(0, 1), 15: geomlib.Axial_Point(1, 1), 16: geomlib.Axial_Point(-2, 2),
                     17: geomlib.Axial_Point(-1, 2), 18: geomlib.Axial_Point(0, 2)}
        return coordDict[hex_i]