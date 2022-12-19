import Classes.Player as Player
import Classes.Board as Board
import Classes.Bank as Bank
import Classes.Move as Move
import random
import time
import math
import os

class Game:
    def __init__(self, num_players = 4):
        self.nplayer = num_players
        self.players = [Player.Player(i+1, self) for i in range(0, num_players)]
        self.largestArmyPlayer = Player.Player(0, self)
        self.longestStreetOwner = Player.Player(0, self)
        self.longestStreetLength = 4
        self.tmpVisitedEdges = []
        self.dice = 0
        self.currentTurn = Player.Player(0, self)
        self.order =0

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
            max = -1
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
        max = -1
        for p in placesToVisit:
            edge = tuple(sorted([p, newPlace]))
            if(Board.Board().edges[edge] == player.id and edge not in self.tmpVisitedEdges):
                self.tmpVisitedEdges.append(edge)
                actual = 1
                if(Board.Board().places[newPlace].owner == 0 or Board.Board().places[newPlace] == player.id): # nuova aggiunta
                    actual += self.longestStreetPlace(player, p, newPlace)
                if(max < actual):
                    max = actual
        return max
              
    def longestStreet(self, player, edge):
        #assert(edge is None, "Edge is None, something went wrong.")    
        #print("Edge:", edge)
        if(edge != None):
            self.tmpVisitedEdges.append(edge)
            toRet = 1 + self.longestStreetPlace(player, edge[0], edge[1]) + self.longestStreetPlace(player, edge[1], edge[0])
        else:
            toRet = 0
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
        max = self.longestStreetLength
        belonger = self.longestStreetOwner
        for p in self.players:
            actual = self.longest(p)
            if(max < actual):
                max = actual
                belonger = p
        if(not justCheck):
            if(belonger != self.longestStreetOwner):
                self.longestStreetOwner = belonger
                print("Longest street length: ", max)
            self.longestStreetLength = max
        print("Out, Longest street length: ", max)
        return belonger

    
    def longest(self, player):
        max = 0
        visited = set()
        for tail in self.findLeaves(player):
            self.order = 0
            # print('Tail: ',tail)
            length, tmpVisited = self.explorePlace(player, tail, [])
            visited.update(tmpVisited)
            if max<length:
                max = length

        for edge in player.ownedStreets:
            print(visited)
            if edge not in visited:
                p1, p2 = edge
                length, tmpVisited =self.explorePlace(player, p1, [])
                visited.update(tmpVisited)
                if max < length:
                    max = length
        return max - 1 
        

    def explorePlace(self, player, place, visited):
        # print('ExporePlace: ', place, self.order)
        self.order +=1
        max = 0
        tmpVisited = list.copy(visited)
        # tmpVisited.append(place)
        outVisited = list.copy(visited)

        for adjPlace in self.connectedPlacesToPlace(player, place):
            edge = tuple(sorted([place, adjPlace]))
            if edge not in tmpVisited:
                tmpVisited.append(edge)
                length, v = self.explorePlace(player, adjPlace, tmpVisited)
                outVisited.extend(v)
                if(max<length):
                    max = length

        # print("Back from ", place)
        return max + 1, outVisited

    def findLeaves(self, player):
        toRet = set()
        for edge in player.ownedStreets:
            p1, p2 = edge
            if self.isLeaf(player, p1):
                toRet.add(p1)
            if self.isLeaf(player, p2):
                toRet.add(p2)
        return toRet

    def isLeaf(self, player, place):
        return len(self.connectedPlacesToPlace(player, place))==1 or len(self.connectedPlacesToPlace(player, place))==3
        
    # def connectedOwnedEdges(self, player, edge):
    #     p1, p2 = edge
    #     own1 = Board.Board().places[p1].owner
    #     own2 = Board.Board().places[p2].owner
    #     edges = set()
    #     if(own1 in (0, player.id)):
    #         print('if1')
    #         edges.update(self.connectedEdgesToPlace(player, p1))
    #     if(own2 in (0, player.id)):
    #         print('if2')
    #         edges.update(self.connectedEdgesToPlace(player, p2))
    #     edges.remove(edge)
    #     return edges


    def connectedPlacesToPlace(self, player, place):
        toRet = []
        if Board.Board().places[place].owner in (0, player.id):
            for adjPlace in Board.Board().graph.listOfAdj[place]:
                edge = tuple(sorted([place, adjPlace]))
                if(Board.Board().edges[edge] == player.id):
                    toRet.append(adjPlace)
        return toRet


        


        





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









                        
        