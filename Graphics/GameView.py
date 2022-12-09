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

        self.font_harbors = pygame.font.SysFont('tahoma', 15)
        self.font_robber = pygame.font.SysFont('tahoma', 50)

        self.points = []
        self.pointsCards = []

        self.woods = []
        self.sheeps = []
        self.crops = []
        self.irons = []
        self.clays = []
        self.knights = []

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
        self.screen.blit(self.points[player.id-1], (x, y)) # 5,5
        self.screen.blit(self.pointsCards[player.id-1], (x, y+55))
        self.screen.blit(self.woods[player.id-1], (x, y+95))
        self.screen.blit(self.sheeps[player.id-1], (x, y+115))
        self.screen.blit(self.crops[player.id-1], (x, y+135))
        self.screen.blit(self.irons[player.id-1], (x, y+155))
        self.screen.blit(self.clays[player.id-1], (x, y+175))
        self.screen.blit(self.knights[player.id-1], (x, y+215))


    def setupAndDisplayBoard(self):
        #Draw the sea
        pygame.draw.rect(self.screen, pygame.Color('lightblue'),
                         (0, 0, 1000, 800))
        # Render each tile
        hexLayout = geomlib.Layout(geomlib.layout_pointy, geomlib.Point(80, 80), geomlib.Point(500, 400))
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
            imgPath = os.path.join(self.sourceFileDir, self.imgDict[boardtile.resource])
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

        self.drawRobber()

        self.checkAndDrawStreets()
        self.checkAndDrawPlaces()

        self.bgScoreColor = pygame.Color("grey32")
        score = pygame.Rect(0,0,120,250) 
        self.screen.fill(self.bgScoreColor, score)  #pygame.Color('lightblue')
        pygame.display.update(score)

        score = pygame.Rect(0,550,120,250) 
        self.screen.fill(self.bgScoreColor, score)  #pygame.Color('lightblue')
        pygame.display.update(score)

        score = pygame.Rect(880,0,120,250) 
        self.screen.fill(self.bgScoreColor, score)  #pygame.Color('lightblue')
        pygame.display.update(score)

        score = pygame.Rect(880,550,120,250) 
        self.screen.fill(self.bgScoreColor, score)  #pygame.Color('lightblue')

        self.updateStats()
        self.blit(self.game.players[0], 5, 5)
        self.blit(self.game.players[1], 5, 550)
        self.blit(self.game.players[2], 900, 5)
        self.blit(self.game.players[3], 900, 550)


        pygame.display.update()
        #pygame.display.update(score)
        #time.sleep(0.1)

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
        robberImg = pygame.image.load(self.robberImgPath).convert_alpha()
        if(self.tempRobberTile != Board.Board().robberTile):
            print("drowing robber...")
            robTile = Board.Board().robberTile
            for graphicTile in self.graphicTileList:
                if(graphicTile.index == robTile):
                    robberCoords = (graphicTile.pixelCenter.x, graphicTile.pixelCenter.y -20)
                    self.screen.blit(robberImg, robberCoords)
                elif(self.tempRobberTile != -1 and self.tempRobberTile == graphicTile.index):
                    imgPath = os.path.join(self.sourceFileDir, self.imgDict[graphicTile.resource])
                    image = pygame.image.load(imgPath).convert_alpha()
                    mask = image.copy()
                    mask = pygame.transform.scale(mask, (130, 130))
                    self.screen.blit(mask, (graphicTile.pixelCenter.x - 65, graphicTile.pixelCenter.y - 65))
                    if graphicTile.resource != 'desert':
                        width = 1000
                        pygame.draw.circle(self.screen, pygame.Color("black"), (graphicTile.pixelCenter.x, graphicTile.pixelCenter.y+30), 17, width==0)
                        pygame.draw.circle(self.screen, pygame.Color("white"), (graphicTile.pixelCenter.x, graphicTile.pixelCenter.y+30), 13, width==0)
                        tileNumberText = self.font_resourceSmallest.render(str(graphicTile.number), False, pygame.Color("black"))

                        if(graphicTile.number >= 10):
                            self.screen.blit(tileNumberText, (graphicTile.pixelCenter.x-10, graphicTile.pixelCenter.y+18))
                        else:
                            self.screen.blit(tileNumberText, (graphicTile.pixelCenter.x-5, graphicTile.pixelCenter.y+18))
            self.tempRobberTile = robTile
                

    def getHexCoords(self, hex_i):
        coordDict = {0: geomlib.Axial_Point(0, -2), 1: geomlib.Axial_Point(1, -2), 2: geomlib.Axial_Point(2, -2),
                     3: geomlib.Axial_Point(-1, -1), 4: geomlib.Axial_Point(0, -1), 5: geomlib.Axial_Point(1, -1),
                     6: geomlib.Axial_Point(2, -1), 7: geomlib.Axial_Point(-2, 0), 8: geomlib.Axial_Point(-1, 0), 9: geomlib.Axial_Point(0, 0),
                     10: geomlib.Axial_Point(1, 0), 11: geomlib.Axial_Point(2, 0), 12: geomlib.Axial_Point(-2, 1),
                     13: geomlib.Axial_Point(-1, 1), 14: geomlib.Axial_Point(0, 1), 15: geomlib.Axial_Point(1, 1), 16: geomlib.Axial_Point(-2, 2),
                     17: geomlib.Axial_Point(-1, 2), 18: geomlib.Axial_Point(0, 2)}
        return coordDict[hex_i]