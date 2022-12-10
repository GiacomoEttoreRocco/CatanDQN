import Classes.Player as Player
import Classes.Board as Board
import Classes.Bank as Bank
import Classes.Move as Move
import random
import time

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
                            #print(self.players[Board.Board().places[p].owner-1].id, "TAKEN ", tile.resource, "\n")
                            Bank.Bank().giveResource(self.players[Board.Board().places[p].owner-1], tile.resource)
                        elif(Board.Board().places[p].isCity):
                            #print(self.players[Board.Board().places[p].owner-1].id, "TAKEN 2 ", tile.resource, "\n")
                            Bank.Bank().giveResource(self.players[Board.Board().places[p].owner-1], tile.resource)
                            Bank.Bank().giveResource(self.players[Board.Board().places[p].owner-1], tile.resource)

    def bestMove(self, player: Player, usedCard):
        moves = player.availableMoves(usedCard)
        player.printResources()
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

    def sevenOnDices(self, player: Player):
        for pyr in self.players:
            if(pyr.resourceCount() >= 7):
                nCards = pyr.resourceCount()/2
                while(pyr.resourceCount() <= nCards):
                    eval, card = player.evaluate(Move.discardResource)
                    Move.discardResource(player, card)
        eval, tilePos = player.evaluate(Move.useRobber)
        Move.useRobber(player, tilePos)   

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
                print("BEFORE ROLL DICE: ", Move.useKnight, "\n")
                turnCardUsed = True 
        if(self.checkWon(player)):
            return
        ####################################################################### ROLL DICES #####################################################################   
        dicesValue = self.rollDice()
        ########################################################################################################################################################
        print("Dice value: ", dicesValue, dicesValue == 7)
        if(dicesValue == 7):
            ev, pos = player.evaluate(Move.useRobber)
            print("POS: ", pos)
            Move.useRobber(player, pos)
            print("POS: ", pos)
        else:
            self.dice_production(dicesValue)

        move, thingNeeded = self.bestMove(player, turnCardUsed)
        move(player, thingNeeded)
        print("Player ", player.id, " mossa: ", move, " ")
        if(self.checkWon(player)):
            return
        if(move in Move.cardMoves()):
                turnCardUsed = True
        while(move != Move.passTurn and not self.checkWon(player)): # move è una funzione 
            move, thingNeeded = self.bestMove(player, turnCardUsed)
            move(player, thingNeeded)
            print("Player ", player.id, " mossa: ", move, " ")
            if(move in Move.cardMoves()):
                turnCardUsed = True

    def checkWon(self, player):
        if(player.victoryPoints >= 10):
            #player.printStats()
            print("Il vincitore è il Player ", str(player.id), "!")
            return True
        return False

    def doInitialChoise(self, player: Player, giveResources = False):
        evaluation, colonyChoosen = player.evaluate(Move.placeFreeColony)
        Move.placeFreeColony(player, colonyChoosen)
        if(giveResources):
            for touchedResource in Board.Board().places[colonyChoosen.id].touchedResourses:
                Bank.Bank().giveResource(player, touchedResource)
        print("Initial choise, colony: ", str(colonyChoosen.id))
        evaluation, edgeChoosen = player.evaluate(Move.placeFreeStreet)
        Move.placeFreeStreet(player, edgeChoosen)

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
                return playerTurn
            if(turn % 4 == 0):
                print("========================================== Start of turn: ", str(int(turn/4)), "=========================================================")









                        
        