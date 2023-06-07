import pygame
import pygame_gui
import Graphics.GraphicTile as GraphicTile
import Graphics.GraphicPlace as GraphicPlace
import Graphics.geomlib as geomlib
import Classes.CatanGraph  as cg
import Classes.Board as Board
import Graphics.PlaceCoordinates as pc
import AI.Gnn as Gnn
import os

class GameView:
    def __init__(self, game, controller):
        pygame.init()
        self.width = 1200
        self.moveListWidth = self.width//5
        self.gameWidth = self.width - self.moveListWidth
        self.height = 700
        self.sourceFileDir = os.path.dirname(os.path.abspath(__file__))
        self.robberImgPath = os.path.join(self.sourceFileDir, "imgs/robber.png")
        self.tempRobberTile = -1 # per motivi di efficienza.
        self.game = game
        self.controller = controller
        windowSize = self.width, self.height
        self.playerColorDict = {0: pygame.Color('grey'), 1: pygame.Color('red'), 2: pygame.Color('yellow'),
                           3: pygame.Color('blueviolet'), 4:  pygame.Color('blue')}
        self.tileColorDict = {"clay": (188, 74, 60), "iron": (128, 128, 128), "crop": pygame.Color('orange'), "wood": (0, 153, 0),
                         "sheep": (51, 255, 51), "desert": (245, 222, 179) }
        self.imgDict = {"clay": "imgs/clay.png", "iron": "imgs/iron.png", "crop": "imgs/crop.png", "wood": "imgs/wood.png",
                   "sheep": "imgs/sheep.png", "desert": "imgs/desert.png", "2:1 wood": "imgs/harbors/21wood.png", 
                   "2:1 crop":"imgs/harbors/21crop.png", "2:1 sheep" : "imgs/harbors/21sheep.png", "2:1 iron":"imgs/harbors/21iron.png", 
                   "2:1 clay":"imgs/harbors/21clay.png", "3:1":"imgs/harbors/31.png"}
        self.graphicTileList = []
        self.graphicPlaceList = []
        self.screen = pygame.display.set_mode(windowSize)

        self.manager = pygame_gui.UIManager(windowSize)
        
        self.priorityButton = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((self.gameWidth/2 - self.height//10 * 4, self.height - self.height//20), (self.height//5, self.height//20)),
                                             text='Priority Mode',
                                             manager=self.manager)
        self.hybridButton = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((self.gameWidth/2 - self.height//10 * 2, self.height - self.height//20), (self.height//5, self.height//20)),
                                             text='Hybrid Mode',
                                             manager=self.manager)
        self.pureButton = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((self.gameWidth/2 - self.height//10  * 0, self.height - self.height//20), (self.height//5, self.height//20)),
                                             text='Pure AI Mode',
                                             manager=self.manager)
        self.rlButton = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((self.gameWidth/2 - self.height//10  * -2, self.height - self.height//20), (self.height//5, self.height//20)),
                                             text='RL Mode',
                                             manager=self.manager)
        self.undoButton = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((self.gameWidth/2 - self.height//10  * 4, self.height - self.height//10), (self.height//5, self.height//20)),
                                             text='UNDO',
                                             manager=self.manager)
        self.redoButton = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((self.gameWidth/2 - self.height//10  * 2, self.height - self.height//10), (self.height//5, self.height//20)),
                                             text='REDO',
                                             manager=self.manager)

        self.stack = pygame_gui.elements.UISelectionList(relative_rect=pygame.Rect((self.gameWidth,0),(self.moveListWidth,self.height)),
                                            item_list=[("player.id", "action")],
                                            manager=self.manager)
        # self.stack = pygame_gui.elements.UITextBox(relative_rect=pygame.Rect((self.gameWidth,0),(self.moveListWidth,self.height)),
        #                                      html_text='REDO',
        #                                      manager=self.manager)

        self.font_resourceLarge = pygame.font.SysFont('tahoma', self.height//18)
        self.font_resourceMedium = pygame.font.SysFont('tahoma', self.height//23)
        self.font_resourceSmaller = pygame.font.SysFont('tahoma', self.height//28)
        self.font_resourceSmallest = pygame.font.SysFont('tahoma', self.height//58, bold=True)

        self.points = []
        self.pointsCards = []
        self.woods = []
        self.sheeps = []
        self.crops = []
        self.irons = []
        self.clays = []
        self.knights = []

        self.monopolyCards = []
        self.roadBuildingCards = []
        self.yearOfPlentyCards = []
        self.playerType = []
        self.bgScoreColor = pygame.Color("grey18")
        self.bgScoreColorHighlited = pygame.Color('grey64')

        for i in range(0, len(self.game.players)):
            self.points.append(self.font_resourceLarge.render(str(self.game.players[i].victoryPoints), False, self.playerColorDict[i+1]))
            self.pointsCards.append(self.font_resourceSmallest.render("Vp cards: " + str(self.game.players[i].victoryPointsCards), False, self.playerColorDict[i+1]))
            self.woods.append(self.font_resourceSmallest.render("Wood: " + str(self.game.players[i].resources["wood"]), False, self.playerColorDict[i+1]))
            self.sheeps.append(self.font_resourceSmallest.render("Sheep: " +str(self.game.players[i].resources["sheep"]), False, self.playerColorDict[i+1]))
            self.crops.append(self.font_resourceSmallest.render("Crop: " + str(self.game.players[i].resources["crop"]), False, self.playerColorDict[i+1]))
            self.irons.append(self.font_resourceSmallest.render("Iron: " + str(self.game.players[i].resources["iron"]), False, self.playerColorDict[i+1]))
            self.clays.append(self.font_resourceSmallest.render("Clay: " + str(self.game.players[i].resources["clay"]), False, self.playerColorDict[i+1]))
            self.knights.append(self.font_resourceSmallest.render("Knights: " + str(self.game.players[i].usedKnights), False, self.playerColorDict[i+1]))
            self.monopolyCards.append(self.font_resourceSmallest.render("MonopolyCards: " + str(self.game.players[i].monopolyCard), False, self.playerColorDict[i+1]))
            self.roadBuildingCards.append(self.font_resourceSmallest.render("RoadBuildingCards: " + str(self.game.players[i].roadBuildingCard), False, self.playerColorDict[i+1]))
            self.yearOfPlentyCards.append(self.font_resourceSmallest.render("YearOfPlentyCards: " + str(self.game.players[i].yearOfPlentyCard), False, self.playerColorDict[i+1]))
            self.playerType.append(self.font_resourceSmallest.render("Type: " + str(self.game.players[i].strategy.name()), False, self.playerColorDict[i+1]))

    def updateStats(self):
        for i in range(0, len(self.game.players)):
            self.points[i] = self.font_resourceLarge.render(str(self.game.players[i].victoryPoints), False, self.playerColorDict[i+1])
            self.pointsCards[i] = self.font_resourceSmallest.render("Vp cards: " + str(self.game.players[i].victoryPointsCards), False, self.playerColorDict[i+1])
            self.woods[i] = self.font_resourceSmallest.render("Wood: " + str(self.game.players[i].resources["wood"]), False, self.playerColorDict[i+1])
            self.sheeps[i] = self.font_resourceSmallest.render("Sheep: " +str(self.game.players[i].resources["sheep"]), False, self.playerColorDict[i+1])
            self.crops[i] = self.font_resourceSmallest.render("Crop: " + str(self.game.players[i].resources["crop"]), False, self.playerColorDict[i+1])
            self.irons[i] = self.font_resourceSmallest.render("Iron: " + str(self.game.players[i].resources["iron"]), False, self.playerColorDict[i+1])
            self.clays[i] = self.font_resourceSmallest.render("Clay: " + str(self.game.players[i].resources["clay"]), False, self.playerColorDict[i+1])
            self.knights[i] = (self.font_resourceSmallest.render("Knights: " + str(self.game.players[i].usedKnights), False, self.playerColorDict[i+1]))
            self.monopolyCards[i] = self.font_resourceSmallest.render("Monopoly: " + str(self.game.players[i].monopolyCard), False, self.playerColorDict[i+1])
            self.roadBuildingCards[i] = self.font_resourceSmallest.render("RoadBuilding: " + str(self.game.players[i].roadBuildingCard), False, self.playerColorDict[i+1])
            self.yearOfPlentyCards[i] = self.font_resourceSmallest.render("YearOfPlenty: " + str(self.game.players[i].yearOfPlentyCard), False, self.playerColorDict[i+1])
            self.playerType[i] = self.font_resourceSmallest.render("Type: " + str(self.game.players[i].strategy.name()), False, self.playerColorDict[i+1])

    def blit(self, player, x, y):
        playerBox = pygame.Rect(x-self.height//200, y-self.height//200, self.height//6.7, self.height//2.5)
        # if self.game.currentTurnPlayer == player:
        #     self.screen.fill(self.bgScoreColorHighlited, playerBox)
        # else:
        #     self.screen.fill(self.bgScoreColor, playerBox)
        self.screen.fill(self.bgScoreColor, playerBox)

        self.screen.blit(self.points[player.id-1], (x, y)) # 5,5
        self.screen.blit(self.pointsCards[player.id-1], (x, y+self.height//18.2))
        self.screen.blit(self.woods[player.id-1], (x, y+self.height//10.5))
        self.screen.blit(self.sheeps[player.id-1], (x, y+self.height//8.7))
        self.screen.blit(self.crops[player.id-1], (x, y+self.height//7.4))
        self.screen.blit(self.irons[player.id-1], (x, y+self.height//6.45))
        self.screen.blit(self.clays[player.id-1], (x, y+self.height//5.7))
        self.screen.blit(self.knights[player.id-1], (x, y+self.height//4.65))
        self.screen.blit(self.monopolyCards[player.id-1], (x, y+self.height//4.2))
        self.screen.blit(self.roadBuildingCards[player.id-1], (x, y+self.height//3.6))
        self.screen.blit(self.yearOfPlentyCards[player.id-1], (x, y+self.height//3.3))
        self.screen.blit(self.playerType[player.id-1], (x, y+self.height//2.8))

    def setupAndDisplayBoard(self, bg = True):
        self.graphicTileList = [] # recently added
        if(bg):
            pygame.draw.rect(self.screen, pygame.Color('cadetblue1'),(0, 0, self.gameWidth, self.height))
        hex_i = 0
        for boardtile in Board.Board().tiles:
            hexCoords = self.getHexCoords(hex_i)
            graphicTile = GraphicTile.GraphicTile(hexCoords, boardtile)
            self.graphicTileList.append(graphicTile)
            self.drawGraphicTile(graphicTile)
            hex_i += 1
        return None

    def drawGraphicTile(self, graphicTile):
        hexLayout = geomlib.Layout(geomlib.layout_pointy, geomlib.Point(self.height//12, self.height//12), geomlib.Point(self.gameWidth/2, self.height/2))
        hexTileCorners = geomlib.polygon_corners(hexLayout, graphicTile.hex)
        tileColorRGB = self.tileColorDict[graphicTile.resource]
        pygame.draw.polygon(self.screen, pygame.Color(tileColorRGB[0], tileColorRGB[1], tileColorRGB[2]),
                            hexTileCorners, self.gameWidth == 0)
        pygame.draw.polygon(self.screen, pygame.Color('black'), hexTileCorners, self.height//200)
        graphicTile.pixelCenter = geomlib.hex_to_pixel(hexLayout, graphicTile.hex)
        imgPath = os.path.join(self.sourceFileDir, self.imgDict[graphicTile.resource])
        image = pygame.image.load(imgPath).convert_alpha()
        mask = image.copy()
        mask = pygame.transform.scale(mask, (self.height//18.5, self.height//18.5))
        self.screen.blit(mask, (graphicTile.pixelCenter.x - self.height//37, graphicTile.pixelCenter.y - self.height//17))
        self.drawNumberCircle(graphicTile)

    def drawNumberCircle(self, graphicTile):
        tileNumberText = self.font_resourceSmaller.render(str(graphicTile.number), False, pygame.Color("black"))
        if graphicTile.resource != 'desert':
            pygame.draw.circle(self.screen, pygame.Color("black"), (graphicTile.pixelCenter.x, graphicTile.pixelCenter.y+(self.height//40)), self.height//37, self.gameWidth==0)
            pygame.draw.circle(self.screen, pygame.Color("white"), (graphicTile.pixelCenter.x, graphicTile.pixelCenter.y+(self.height//40)), self.height//43, self.gameWidth==0)
            if(graphicTile.number >= 10):
                self.screen.blit(tileNumberText, (graphicTile.pixelCenter.x-self.height//50, graphicTile.pixelCenter.y))
            else:
                self.screen.blit(tileNumberText, (graphicTile.pixelCenter.x-self.height//100, graphicTile.pixelCenter.y))

    def setupPlaces(self):
        self.graphicPlaceList = [] # recently added
        for place in Board.Board().places:
            self.graphicPlaceList.append(GraphicPlace.GraphicPlace(place))
        alreadyFound = []
        for gtile in self.graphicTileList:
            gtile.places = [] # recently added
            for k, v in cg.tilePlaces.items():
                if gtile.index == k:
                    for el in v:
                        if el not in alreadyFound:
                            placeToAdd = self.graphicPlaceList[el]
                            placeToAdd.setupCoords(pc.getCoords(self, placeToAdd.index))
                            placeToAdd.setupSize((self.height // 20, self.height // 20))
                            gtile.places.append(placeToAdd)
                            self.checkAndDrawHarbors(placeToAdd)
                            alreadyFound.append(el)

    def updateScoreGNN(self, player):
        scores_string = "Scores: "
        normalizedScores = [0,0,0,0]
        sum = 0
        for i, evaluatePlayer in enumerate(player.game.players):
            normalizedScores[i] = Gnn.Gnn().evaluatePositionForPlayer(evaluatePlayer)
            sum += normalizedScores[i]

        # scores_string += '%.2f '% Gnn.Gnn().evaluatePositionForPlayer(evaluatePlayer)
        for i, evaluatePlayer in enumerate(player.game.players):
            if(sum == 0):
                scores_string += '%.2f '% 0 #normalizedScores[i]/sum
            else:
                scores_string += '%.2f '% (normalizedScores[i]/sum)

        # scores = self.font_resourceSmallest.render(scores_string, False, self.playerColorDict[0]) # self.playerColorDict[player.id])
        scores = self.font_resourceMedium.render(scores_string, False, pygame.Color('grey')) # self.playerColorDict[player.id])

        # playersScores = pygame.Rect(self.gameWidth/2-self.height//8, 0, self.height//4, self.height//20)
        # self.screen.fill(self.bgScoreColor, playersScores)
        # self.screen.blit(scores, (self.gameWidth/2-self.height//8.3, self.height//200))

        playersScores = pygame.Rect(self.gameWidth/2-self.height//4, 0, self.height//1.8, self.height//11)
        self.screen.fill(self.bgScoreColor, playersScores)
        self.screen.blit(scores, (self.gameWidth/2-self.height//4.3, self.height//100))

    def updateGameScreen(self, flag=False):
        self.setupAndDisplayBoard() # recently added
        self.setupPlaces()
        self.drawRobber()
        self.checkAndDrawStreets()
        self.checkAndDrawPlaces()
        self.updateStats()
        self.updateScoreGNN(self.game.currentTurnPlayer)

        if(2 <= self.game.nplayers):
            self.blit(self.game.players[0], self.height//200, self.height//200)
            self.blit(self.game.players[1], self.gameWidth-self.height//6.9, self.height//200)
        if(3 <= self.game.nplayers):
            self.blit(self.game.players[2], self.height//200, self.height-self.height//2.6)
        if(4 <= self.game.nplayers):
            self.blit(self.game.players[3], self.gameWidth-self.height//6.9, self.height-self.height//2.6)

        longestStreetBox = pygame.Rect(self.height//5.8, self.height-self.height//10, self.gameWidth * 0.1, self.height//10)
        self.screen.fill(self.bgScoreColor, longestStreetBox)
        font_longest_street = self.font_resourceSmaller.render('LS: '+ str(self.game.longestStreetOwner.id), False, pygame.Color('white'))
        self.screen.blit(font_longest_street, (self.height//5.7, self.height - self.height//10.5))
        font_longest_street_length = self.font_resourceSmaller.render('len: '+ str(self.game.longestStreetLength), False, pygame.Color('white'))
        self.screen.blit(font_longest_street_length, (self.height//5.7, self.height - self.height//22.2))
        
        largestArmyBox = pygame.Rect(self.gameWidth-self.height//3.7, self.height-self.height//20, self.height//10, self.height//20)
        self.screen.fill(self.bgScoreColor, largestArmyBox)
        font_largest_army = self.font_resourceSmaller.render('LA: '+ str(self.game.largestArmyPlayer.id), False, pygame.Color('white'))
        self.screen.blit(font_largest_army, (self.gameWidth-self.height//3.77, self.height - self.height//22.2))

        font_dice = self.font_resourceSmaller.render(str(self.game.dices[self.game.actualTurn-1]), False, pygame.Color('white'))
        diceRoll = pygame.Rect(self.height//5.8, 0, self.height//20, self.height//20)
        self.screen.fill(self.bgScoreColor, diceRoll)
        self.screen.blit(font_dice, (self.height//5.7, self.height//200))

        undoStack = self.controller.summaryUndoStack()
        redoStack = self.controller.summaryRedoStack()
        
        if not flag:
            self.stack.set_item_list(["MOVES LIST"])
            self.stack.add_items(redoStack)
            self.stack.add_items(reversed(undoStack))
            if len(self.stack.item_list) > 1 and len(undoStack)>0 and self.stack.item_list[len(redoStack)+1]['button_element'] != None:
                pygame.draw.rect(self.stack.item_list[len(redoStack)+1]['button_element'].image, pygame.Color('red'), [0, 0, self.moveListWidth-self.height//140, self.height//35], 2)

        self.manager.update(0)
        self.manager.draw_ui(self.screen)

        pygame.display.update()

    def drawPlace(self, graphicPlace):
        graphicPlace.setupSprite()
        sprList = pygame.sprite.Group()
        sprList.add(graphicPlace.sprite)
        sprList.draw(self.screen)

    def drawStreet(self, edge, color):
        startPos = edge[0]
        endPos = edge[1]
        pygame.draw.line(self.screen, pygame.Color("Black"), self.graphicPlaceList[startPos].coords, self.graphicPlaceList[endPos].coords, self.height//50)
        pygame.draw.line(self.screen, color, self.graphicPlaceList[startPos].coords, self.graphicPlaceList[endPos].coords, self.height//100)

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

    def checkAndDrawHarbors(self, graphicPlace):
        idPlace = graphicPlace.index
        if graphicPlace.harbor is not None:
            harborPath = os.path.join(self.sourceFileDir, self.imgDict[graphicPlace.harbor])
            harborImg = pygame.transform.scale(pygame.image.load(harborPath).convert_alpha(),(self.height//20, self.height//20))

            placesOnRightUp = [6,14,15,25,26]
            placesOnRightDown = [37,36,46,45,53]

            placesOnLeftUp = [0,8,7,17,16]
            placesOnLeftDown = [27,28,38,39,47]

            placesOnTop = [1,2,3,4,5]
            placesOnDown = [48,49,50,51,52]

            if(idPlace in placesOnLeftUp):
                coords = (graphicPlace.coords[0]-self.height//18, graphicPlace.coords[1]-self.height//18)
                self.screen.blit(harborImg, coords)
            elif(idPlace in placesOnLeftDown):
                coords = (graphicPlace.coords[0]-self.height//18, graphicPlace.coords[1]+self.height//200)
                self.screen.blit(harborImg, coords)
            elif(idPlace in placesOnRightUp):
                coords = (graphicPlace.coords[0]+self.height//200, graphicPlace.coords[1]-self.height//18)
                self.screen.blit(harborImg, coords)
            elif(idPlace in placesOnRightDown):
                coords = (graphicPlace.coords[0]+self.height//200, graphicPlace.coords[1]+self.height//200)
                self.screen.blit(harborImg, coords)
            elif(idPlace in placesOnTop):
                coords = (graphicPlace.coords[0]-self.height//40, graphicPlace.coords[1]-self.height//15)
                self.screen.blit(harborImg, coords)
            elif(idPlace in placesOnDown):
                coords = (graphicPlace.coords[0]-self.height//40, graphicPlace.coords[1]+self.height//66)
                self.screen.blit(harborImg, coords)

    def drawRobber(self):
        robberImg = pygame.image.load(self.robberImgPath).convert_alpha()
        robberImg = pygame.transform.scale(robberImg, (self.height//15, self.height//15))
        robTile = Board.Board().robberTile
        for graphicTile in self.graphicTileList:
            if(graphicTile.index == robTile):
                robberCoords = (graphicTile.pixelCenter.x, graphicTile.pixelCenter.y-self.height//33)
                self.screen.blit(robberImg, robberCoords)
            elif(self.tempRobberTile != -1 and self.tempRobberTile == graphicTile.index):
                self.drawGraphicTile(graphicTile)
        self.tempRobberTile = robTile           

    def getHexCoords(self, hex_i):
        coordDict = {0: geomlib.Axial_Point(0, -2), 1: geomlib.Axial_Point(1, -2), 2: geomlib.Axial_Point(2, -2),
                     3: geomlib.Axial_Point(-1, -1), 4: geomlib.Axial_Point(0, -1), 5: geomlib.Axial_Point(1, -1),
                     6: geomlib.Axial_Point(2, -1), 7: geomlib.Axial_Point(-2, 0), 8: geomlib.Axial_Point(-1, 0), 9: geomlib.Axial_Point(0, 0),
                     10: geomlib.Axial_Point(1, 0), 11: geomlib.Axial_Point(2, 0), 12: geomlib.Axial_Point(-2, 1),
                     13: geomlib.Axial_Point(-1, 1), 14: geomlib.Axial_Point(0, 1), 15: geomlib.Axial_Point(1, 1), 16: geomlib.Axial_Point(-2, 2),
                     17: geomlib.Axial_Point(-1, 2), 18: geomlib.Axial_Point(0, 2)}
        return coordDict[hex_i]