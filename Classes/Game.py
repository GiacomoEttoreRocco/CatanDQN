import Classes.Player as Player
import Classes.Board as Board
import Classes.Bank as Bank
from Classes.Strategy.Strategy import Strategy
import Command.commands as commands
import Command.controller as controller

import random
# import time
import math
import os

class Game:
    def __init__(self, num_players):
        self.ctr = controller.ActionController()
        self.dummy = Player.Player(0, self, Strategy)
        self.dummy.victoryPoints = 4
        self.nplayers = num_players
        self.players = [Player.Player(i+1, self, None) for i in range(0, num_players)]
        self.largestArmyPlayer = self.dummy
        self.longestStreetOwner = self.dummy
        self.longestStreetLength = 4
        self.tmpVisitedEdges = []
        self.dices = [self.rollDice() for _ in range(1000)]
        self.actualTurn = 0
        self.currentTurnPlayer = self.players[0] #self.dummy
        #self.order = 0

        # DEBUG:
        # for i in range(num_players):
        #     assert self.players[i].resourceCount() == 0
        #     assert self.players[i].victoryPoints == 0
        #     assert len(self.players[i].ownedStreets) == 0
        #     assert len(self.players[i].ownedCities) == 0
        #     assert len(self.players[i].ownedColonies) == 0
        #     assert len(self.players[i].ownedHarbors) == 0
        #     assert self.players[i].nCities == 0
        #     assert self.players[i].nColonies == 0
        #     assert self.players[i].nStreets == 0
        #     assert self.players[i].unusedKnights == 0

    def reset(self):
        # self.ctr = controller.ActionController()
        self.dummy = Player.Player(0, self, Strategy)
        self.dummy.victoryPoints = 4
        # self.nplayers = num_players
        for p in self.players:
            p.reset()

        self.largestArmyPlayer = self.dummy
        self.longestStreetOwner = self.dummy
        self.longestStreetLength = 4
        self.tmpVisitedEdges = []
        self.dices = [self.rollDice() for _ in range(1000)]
        self.actualTurn = 0
        self.currentTurnPlayer = self.players[0] #self.dummy

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
            #self.order = 0
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

    def isLeaf(self, player, place): #  Per il Jack del futuro: non è errato, richiede un po' di ragionamento
        return len(self.connectedPlacesToPlace(player, place))==1 or len(self.connectedPlacesToPlace(player, place))==3 # se == 3 allora si trova in un punto ciclico.
    
    def connectedPlacesToPlace(self, player, place):
        toRet = []
        if Board.Board().places[place].owner in (0, player.id):
            for adjPlace in Board.Board().graph.listOfAdj[place]:
                edge = tuple(sorted([place, adjPlace]))
                if(Board.Board().edges[edge] == player.id):
                    toRet.append(adjPlace)
        return toRet   
    
    def getTotalState(self, player):
        state = Board.Board().placesState(player) + player.globalFeaturesState() # da capire come, probabilmente bisogna assemblare i 3 tensori
        return state

    def getBoardState(self, player):
        return Board.Board().placesState(player)
    
    def getPlayerGlobalFeaturesState(self, player):
        return player.globalFeaturesState()
