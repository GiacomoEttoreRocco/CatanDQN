import Classes.PlayerWithCommands as Player
import Classes.Board as Board
import Classes.Bank as Bank
#import Classes.Action as Action
import Command.commands as commands
import Command.controller as controller

import random
import time
import math
import os

class Game:
    def __init__(self, num_players = 4):

        self.ctr = controller.ActionController()
        ########################################## dummy is necessary: debugging and functioning reason. 
        self.dummy = Player.Player(0, self)
        self.dummy.victoryPoints = 4
        ##########################################
        self.nplayer = num_players
        self.players = [Player.Player(i+1, self) for i in range(0, num_players)]
        self.largestArmyPlayer = self.dummy
        self.longestStreetOwner = self.dummy
        self.longestStreetLength = 4
        self.tmpVisitedEdges = []
        self.dices = [self.rollDice() for _ in range(1000)]
        self.actualTurn = 0
        self.currentTurnPlayer = self.players[0] #self.dummy
        self.order = 0

        for i in range(num_players):
            assert self.players[i].resourceCount() == 0
            assert self.players[i].victoryPoints == 0
            assert len(self.players[i].ownedStreets) == 0
            assert len(self.players[i].ownedCities) == 0
            assert len(self.players[i].ownedColonies) == 0
            assert len(self.players[i].ownedHarbors) == 0
            assert self.players[i].nCities == 0
            assert self.players[i].nColonies == 0
            assert self.players[i].nStreets == 0
            assert self.players[i].unusedKnights == 0

    def printVictoryPointsOfAll(self, nDevCardsBought: list):
        for player in self.players:
            s = str(player.id) + " : " + str(player.victoryPoints) + " -> Points from colony: +" + str(player.nColonies) + " From cities: +" + str(player.nCities*2) + " From vpCards: +" + str(player.victoryPointsCards)
            if(player.id == self.largestArmyPlayer.id):
                s += " From Knigths +2 "
            if(player.id == self.longestStreetOwner.id):    
                s += " From Streets +2 "
            s += " Number of DevCards bounght: " + str(nDevCardsBought[player.id-1])
            print(s) 

        s = str(self.dummy.id) + " : " + str(self.dummy.victoryPoints) + " -> Points from colony: +" + str(self.dummy.nColonies) + " From cities: +" + str(self.dummy.nCities*2) + " From vpCards: +" + str(self.dummy.victoryPointsCards)
        if(self.dummy.id == self.largestArmyPlayer.id):
            s += " From Knigths +2 "
        if(self.dummy.id == self.longestStreetOwner.id):    
            s += " From Streets +2 "
        print(s) 

        for player in self.players:
            s = "Player Id: " +str(player.id) + " : "
            assert player.victoryPoints >= 2, s+"\nError: found a player with less then 2 points."
            assert player.victoryPoints <= 12, s+"\nError: found a player with more then 11 points."

        assert self.dummy.victoryPoints >= 0, s+"\nError, fake player weird behaviour: less then 0 points."
        assert self.dummy.victoryPoints <= 4, s+"\nError: fake player weird behaviour: more then 4 points."

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

    def bestAction(self, player: Player):
        if(player.AI or player.RANDOM):
            if(self.actualTurn<4):
                actions = [commands.FirstChoiseCommand]
            elif(self.actualTurn<8):
                actions = [commands.SecondChoiseCommand]
            else:
                actions = player.availableActions(player.turnCardUsed)

            max = -1
            thingsNeeded = None
            bestAction = actions[0]
            for action in actions: 
                evaluation, tempInput = player.evaluate(action)
                if(max <= evaluation):
                    max = evaluation
                    thingsNeeded = tempInput
                    bestAction = action

            onlyPassTurn = commands.PassTurnCommand in actions and len(actions)==1
            return bestAction, thingsNeeded, onlyPassTurn
        # else:
        #     actions = player.availableActionsWithInput(usedCard)
        #     return player.chooseAction(actions)

    # def sevenOnDices(self):
    #     #ctr = controller.ActionController()
    #     for pyr in self.players:
    #         # print("Resource count: ", pyr.resourceCount())
    #         half = int(pyr.resourceCount()/2)
    #         if(pyr.resourceCount() >= 7):
    #             if(pyr.AI or pyr.RANDOM):
    #                 for i in range(0, half):
    #                     eval, resource = pyr.evaluate(commands.DiscardResourceCommand)
    #                     self.ctr.execute(commands.DiscardResourceCommand(pyr, resource))

    def rollDice(self): 
        return random.randint(1,6) + random.randint(1,6)    

    def checkWon(self, player):
        if(player.victoryPoints >= 10):
            return True
        return False

    def totalKnightsUsed(self):
        totKnightUsed = 0
        for p in self.players:
            totKnightUsed = totKnightUsed + p.usedKnights
        return totKnightUsed
    
    def largestArmy(self):
        max = self.largestArmyPlayer.usedKnights
        belonger = self.largestArmyPlayer
        for p in self.players:
            if(p.usedKnights >= 3 and p.usedKnights > max):
                max = p.usedKnights 
                belonger = p
        return belonger

    def longestStreetPlayer(self):
        maxLength = max([self.longest(self.longestStreetOwner), 4])
        if(maxLength > 4):
            belonger = self.longestStreetOwner
        else:
            belonger = self.dummy
        for p in self.players:
            if(p.id != self.longestStreetOwner.id):
                actual = self.longest(p)
                if(maxLength < actual):
                    maxLength = actual
                    belonger = p
        return belonger, maxLength

    def longest(self, player):
        max = 0
        visited = set()
        for tail in self.findLeaves(player):
            self.order = 0
            length, tmpVisited = self.explorePlace(player, tail, [])
            visited.update(tmpVisited)
            if max<length:
                max = length

        for edge in player.ownedStreets:
            if edge not in visited:
                p1, p2 = edge
                length1, tmpVisited1 = self.explorePlace(player, p1, [])
                length2, tmpVisited2 = self.explorePlace(player, p2, []) # for fun
                visited.update(tmpVisited1)
                visited.update(tmpVisited2)
                if max < length1:
                    max = length1
                elif max < length2:
                    max = length2
        return max - 1 
        
    def explorePlace(self, player, place, visited):
        self.order +=1
        max = 0
        tmpVisited = list.copy(visited)
        outVisited = list.copy(visited)
        for adjPlace in self.connectedPlacesToPlace(player, place):
            edge = tuple(sorted([place, adjPlace]))
            if edge not in tmpVisited:
                tmpVisited.append(edge)
                length, v = self.explorePlace(player, adjPlace, tmpVisited)
                outVisited.extend(v)
                if(max<length):
                    max = length
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

    def connectedPlacesToPlace(self, player, place):
        toRet = []
        if Board.Board().places[place].owner in (0, player.id):
            for adjPlace in Board.Board().graph.listOfAdj[place]:
                edge = tuple(sorted([place, adjPlace]))
                if(Board.Board().edges[edge] == player.id):
                    toRet.append(adjPlace)
        return toRet

    # def playGame(self):
    #     Board.Board().reset()
    #     Bank.Bank().reset()
    #     turn = 1 
    #     won = False
    #     for p in self.players:
    #         self.doInitialChoise(p)
    #     for p in sorted(self.players, reverse=True):
    #         self.doInitialChoise(p, giveResources=True)
    #     while won == False:
    #         playerTurn = self.players[turn%self.nplayer]
    #         turn += 1
    #         self.doTurn(playerTurn)
    #         if(playerTurn.victoryPoints >= 10):
    #             print("The winner is: ", playerTurn, "!")
    #             # time.sleep(5)
    #             return playerTurn
    #         if(turn % 4 == 0):
    #             print("========================================== Start of turn: ", str(int(turn/4)), "=========================================================")









                        
        