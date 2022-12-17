import Classes.Player as Player
import Classes.Board as Board
import Classes.Bank as Bank
import Classes.Move as Move
import random
import time
import math

class Game:
    def __init__(self, num_players = 4):
        self.nplayer = num_players
        self.players = [Player.Player(i+1, self) for i in range(0, num_players)]
        self.largestArmyPlayer = Player.Player(0, self)
        self.longestStreetOwner = Player.Player(0, self)
        self.longestStreetLength = 0
        self.tmpVisitedEdges = []
        self.dice = 0
        self.currentTurn = Player.Player(0, self)

    def dice_production(self, number):
        for tile in Board.Board().tiles:
            if tile.number == number and tile != Board.Board().robberTile:
                for p in tile.associatedPlaces:
                    if(Board.Board().places[p].owner != 0):
                        if(Board.Board().places[p].isColony):
                            Bank.Bank().giveResource(self.players[Board.Board().places[p].owner-1], tile.resource)
                        elif(Board.Board().places[p].isCity):
                            Bank.Bank().giveResource(self.players[Board.Board().places[p].owner-1], tile.resource)
                            Bank.Bank().giveResource(self.players[Board.Board().places[p].owner-1], tile.resource)

    def bestMove(self, player: Player, usedCard):
        if(player.AI or player.RANDOM):
            moves = player.availableMoves(usedCard)
            # player.printResources()
            max = 0
            thingsNeeded = None
            bestMove = Move.passTurn
            for move in moves:
                evaluation, tempInput = player.evaluate(move)
                if(max <= evaluation):
                    max = evaluation
                    thingsNeeded = tempInput
                    bestMove = move
            return bestMove, thingsNeeded
        else:
            moves = player.availableMovesWithInput(usedCard)
            return player.chooseMove(moves)

    def sevenOnDices(self):
        for pyr in self.players:
            # print("Resource count: ", pyr.resourceCount())
            half = int(pyr.resourceCount()/2)
            if(pyr.resourceCount() >= 7):
                if(pyr.AI or pyr.RANDOM):
                    for i in range(0, half):
                        eval, card = pyr.evaluate(Move.discardResource)
                        Move.discardResource(pyr, card)
                else:
                    for i in range(0, half):
                        moves = []
                        for res in pyr.resources.keys():
                            if(pyr.resources[res] > 0):
                                moves.append((Move.discardResource, res))
                        move, resource = pyr.chooseMove(moves)
                        Move.discardResource(pyr, resource)

    def rollDice(self): 
        return random.randint(1,6) + random.randint(1,6)    

    def doTurn(self, player: Player):
        turnCardUsed = False
        player.unusedKnights = player.unusedKnights + player.justBoughtKnights
        player.justBoughtKnights = 0
        player.monopolyCard += player.justBoughtMonopolyCard
        player.justBoughtMonopolyCard = 0
        player.roadBuildingCard += player.justBoughtRoadBuildingCard
        player.justBoughtRoadBuildingCard = 0
        player.yearOfPlentyCard += player.justBoughtYearOfPlentyCard
        player.justBoughtYearOfPlentyCard = 0
        if(player.unusedKnights > 0 and not turnCardUsed):
            actualEvaluation = Board.Board().actualEvaluation()
            afterKnightEvaluation, place = player.evaluate(Move.useKnight)
            if(afterKnightEvaluation > actualEvaluation):
                Move.useKnight(player, place)
                # print("BEFORE ROLL DICE: ", Move.useKnight, "\n")
                turnCardUsed = True 
        if(self.checkWon(player)):
            return
        ####################################################################### ROLL DICES #####################################################################   
        dicesValue = self.rollDice()
        ########################################################################################################################################################
        if(dicesValue == 7):
            self.sevenOnDices(player)
            ev, pos = player.evaluate(Move.useRobber)
            Move.useRobber(player, pos)
        else:
            self.dice_production(dicesValue)

        move, thingNeeded = self.bestMove(player, turnCardUsed)
        move(player, thingNeeded)
        if(self.checkWon(player)):
            return
        if(move in Move.cardMoves()):
                turnCardUsed = True
        while(move != Move.passTurn and not self.checkWon(player)): # move è una funzione 
            move, thingNeeded = self.bestMove(player, turnCardUsed)
            move(player, thingNeeded)
            if(move in Move.cardMoves()):
                turnCardUsed = True

    def checkWon(self, player):
        if(player.victoryPoints >= 10):
            return True
        return False

    def doInitialChoise(self, player: Player, giveResources = False):
        if(player.AI or player.RANDOM):
            evaluation, colonyChoosen = player.evaluate(Move.placeInitialColony)
            Move.placeInitialColony(player, colonyChoosen)
            if(giveResources):
                for touchedResource in Board.Board().places[colonyChoosen.id].touchedResourses:
                    Bank.Bank().giveResource(player, touchedResource)
            # print("Initial choise, colony: ", str(colonyChoosen.id))
            evaluation, edgeChoosen = player.evaluate(Move.placeInitialStreet)
            Move.placeInitialStreet(player, edgeChoosen)
        else:
            moves = []
            # print("DBUG: ", player.calculatePossibleInitialColony())
            for colony in player.calculatePossibleInitialColony():
                moves.append((Move.placeInitialColony, colony))
            move, colonyChoosen = player.chooseMove(moves)
            Move.placeInitialColony(player, colonyChoosen)
            if(giveResources):
                for touchedResource in Board.Board().places[colonyChoosen.id].touchedResourses:
                    Bank.Bank().giveResource(player, touchedResource)
            moves = []
            for street in player.calculatePossibleInitialStreets():
                moves.append((Move.placeInitialStreet, street))
            move, edgeChoosen = player.chooseMove(moves)
            Move.placeInitialStreet(player, edgeChoosen)

    def totalKnightsUsed(self):
        totKnightUsed = 0
        for p in self.players:
            totKnightUsed = totKnightUsed + p.usedKnights
        return totKnightUsed
    
    def largestArmy(self, justCheck = False):
        max = self.largestArmyPlayer.usedKnights
        belonger = self.largestArmyPlayer
        for p in self.players:
            if(p.usedKnights >= 3 and p.usedKnights > max):
                max = p.usedKnights 
                belonger = p
        if(not justCheck and belonger.id != 0):
            self.largestArmyPlayer = self.players[belonger.id-1]
        return belonger

    def longestStreetPlace(self, player, newPlace, oldPlace):
        placesToVisit = list.copy(Board.Board().graph.listOfAdj[newPlace])
        placesToVisit.remove(oldPlace)
        max = 0
        for p in placesToVisit:
            edge = tuple(sorted([p, newPlace]))
            if(Board.Board().edges[edge] == player.id and edge not in self.tmpVisitedEdges):
                self.tmpVisitedEdges.append(edge)
                actual = 1 + self.longestStreetPlace(player, p, newPlace)
                if(max < actual):
                    max = actual
        return max
              
    def longestStreet(self, player, edge):
        assert(edge is None, "Edge is None, something went wrong.")    
        self.tmpVisitedEdges.append(edge)
        toRet =  1 + self.longestStreetPlace(player, edge[0], edge[1]) + self.longestStreetPlace(player, edge[1], edge[0])
        return toRet

    def calculateLongestStreet(self, player):
        self.tmpVisitedEdges = []
        max = 0
        for edge in player.ownedStreets:
            if edge not in self.tmpVisitedEdges:
                actual = self.longestStreet(player, edge)
                if(max < actual):
                    max = actual
        return max

    def longestStreetPlayer(self, justCheck = False):
        max = 4 
        belonger = self.longestStreetOwner
        for p in self.players:
            actual = self.calculateLongestStreet(p)
            if(max < actual):
                max = actual
                belonger = p
        if(not justCheck):
            if(belonger != self.longestStreetOwner):
                self.longestStreetOwner = belonger
                self.longestStreetLength = max
        return belonger

    def playGame(self):
        Board.Board().reset()
        Bank.Bank().reset()
        turn = 1 
        won = False
        for p in self.players:
            self.doInitialChoise(p)
        for p in sorted(self.players, reverse=True):
            self.doInitialChoise(p, giveResources=True)
        while won == False:
            playerTurn = self.players[turn%self.nplayer]
            turn += 1
            self.doTurn(playerTurn)
            if(playerTurn.victoryPoints >= 10):
                print("The winner is: ", playerTurn, "!")
                # time.sleep(5)
                return playerTurn
            if(turn % 4 == 0):
                print("========================================== Start of turn: ", str(int(turn/4)), "=========================================================")









                        
        