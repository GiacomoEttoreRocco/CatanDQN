import torch
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
        # self.dices = [self.rollDice() for _ in range(1200)]
        self.dices = [4, 9, 7, 8, 6, 9, 6, 3, 6, 5, 6, 6, 10, 7, 7, 12, 11, 2, 5, 7, 5, 8, 9, 8, 9, 6, 9, 4, 4, 5, 10, 4, 8, 7, 7, 3, 6, 2, 10, 5, 6, 9, 8, 10, 7, 5, 6, 5, 7, 3, 8, 7, 8, 4, 8, 6, 2, 10, 12, 8, 7, 10, 5, 7, 9, 10, 7, 8, 8, 11, 6, 5, 7, 6, 8, 10, 9, 9, 7, 12, 5, 9, 7, 7, 7, 7, 10, 4, 11, 6, 5, 5, 9, 11, 11, 6, 4, 6, 10, 12, 8, 8, 6, 4, 3, 6, 7, 7, 4, 6, 7, 5, 9, 11, 4, 11, 6, 6, 8, 4, 4, 6, 8, 5, 10, 7, 7, 5, 6, 5, 2, 10, 11, 5, 9, 6, 7, 7, 5, 9, 7, 11, 7, 4, 9, 7, 7, 5, 10, 9, 4, 8, 9, 8, 10, 2, 10, 2, 6, 3, 10, 4, 4, 9, 8, 10, 8, 6, 6, 3, 10, 7, 4, 8, 7, 7, 3, 10, 7, 10, 6, 8, 8, 11, 12, 10, 9, 8, 6, 9, 6, 5, 7, 5, 8, 10, 8, 7, 6, 9, 5, 6, 7, 5, 5, 9, 6, 6, 3, 4, 11, 8, 6, 2, 9, 6, 7, 8, 8, 6, 11, 8, 8, 7, 11, 8, 9, 4, 4, 6, 4, 11, 7, 8, 3, 3, 8, 7, 3, 6, 4, 8, 12, 6, 11, 7, 4, 5, 8, 5, 6, 3, 8, 4, 6, 7, 8, 9, 9, 10, 9, 9, 2, 9, 7, 8, 10, 10, 4, 7, 9, 3, 11, 6, 9, 6, 6, 12, 11, 8, 11, 5, 7, 8, 10, 11, 8, 7, 2, 6, 8, 6, 5, 7, 2, 11, 8, 10, 9, 5, 7, 3, 6, 7, 6, 6, 7, 9, 3, 8, 3, 5, 9, 6, 7, 9, 8, 5, 4, 8, 8, 10, 8, 10, 7, 7, 9, 8, 9, 6, 5, 8, 4, 5, 5, 5, 7, 6, 7, 6, 10, 9, 10, 4, 10, 6, 7, 5, 4, 7, 5, 9, 6, 4, 8, 6, 5, 8, 6, 10, 8, 3, 6, 10, 8, 6, 8, 11, 5, 4, 9, 11, 6, 10, 6, 4, 8, 5, 8, 6, 3, 7, 5, 10, 7, 6, 8, 9, 7, 11, 8, 8, 4, 7, 10, 10, 4, 6, 8, 6, 10, 12, 5, 2, 5, 4, 6, 6, 10, 11, 3, 7, 8, 5, 7, 6, 10, 4, 8, 7, 8, 6, 5, 2, 5, 11, 5, 2, 11, 6, 8, 10, 3, 7, 8, 6, 3, 7, 8, 6, 10, 3, 4, 10, 11, 8, 4, 7, 4, 9, 3, 11, 2, 6, 8, 2, 8, 7, 8, 12, 2, 7, 4, 4, 12, 9, 5, 7, 9, 4, 7, 8, 5, 4, 11, 8, 9, 7, 5, 8, 4, 8, 11, 6, 10, 9, 8, 7, 5, 6, 5, 8, 8, 10, 7, 8, 7, 8, 5, 4, 12, 8, 3, 7, 4, 11, 7, 5, 9, 8, 5, 9, 6, 8, 7, 8, 7, 7, 7, 4, 8, 9, 6, 7, 7, 2, 8, 6, 8, 7, 3, 6, 10, 5, 9, 6, 8, 6, 8, 5, 5, 4, 4, 7, 6, 7, 4, 7, 9, 6, 3, 8, 9, 10, 8, 7, 8, 11, 5, 2, 6, 6, 10, 8, 7, 8, 3, 6, 4, 12, 3, 9, 9, 3, 7, 11, 10, 5, 9, 4, 4, 7, 3, 6, 11, 5, 11, 6, 8, 4, 4, 10, 9, 11, 5, 7, 3, 7, 8, 7, 7, 12, 7, 7, 5, 9, 9, 3, 4, 3, 6, 8, 5, 9, 7, 5, 6, 7, 8, 5, 11, 8, 7, 8, 7, 5, 6, 5, 3, 2, 8, 7, 7, 8, 9, 11, 9, 7, 3, 8, 10, 2, 7, 10, 2, 8, 8, 6, 8, 10, 4, 7, 3, 4, 7, 6, 8, 2, 4, 2, 6, 5, 6, 9, 8, 9, 7, 5, 2, 10, 7, 7, 12, 10, 6, 4, 5, 6, 7, 9, 3, 9, 9, 6, 2, 8, 7, 6, 6, 3, 8, 6, 7, 8, 9, 5, 5, 5, 7, 6, 6, 5, 8, 9, 9, 3, 7, 9, 6, 7, 6, 3, 10, 3, 8, 10, 6, 8, 6, 5, 3, 6, 6, 2, 7, 4, 5, 9, 10, 4, 12, 6, 4, 5, 5, 7, 7, 5, 2, 6, 4, 5, 3, 11, 7, 8, 4, 8, 6, 2, 3, 8, 12, 8, 7, 5, 8, 11, 8, 8, 6, 7, 7, 10, 12, 9, 8, 7, 8, 8, 3, 4, 4, 5, 5, 8, 10, 6, 5, 8, 9, 9, 8, 6, 7, 4, 11, 6, 10, 9, 5, 7, 6, 9, 7, 9, 5, 3, 11, 7, 10, 5, 8, 4, 4, 10, 5, 6, 7, 4, 6, 6, 4, 10, 11, 4, 5, 5, 5, 7, 9, 3, 10, 10, 6, 12, 2, 9, 4, 6, 9, 5, 7, 5, 3, 6, 7, 6, 11, 4, 7, 5, 2, 10, 6, 9, 12, 8, 8, 7, 11, 5, 8, 7, 8, 11, 9, 12, 7, 6, 4, 7, 7, 2, 5, 5, 9, 9, 8, 5, 8, 9, 6, 9, 10, 5, 4, 5, 4, 5, 6, 5, 6, 3, 8, 6, 7, 6, 6, 4, 8, 2, 7, 4, 5, 9, 5, 7, 11, 6, 4, 11, 4, 7, 9, 4, 9, 8, 6, 8, 5, 7, 6, 7, 9, 9, 10, 12, 7, 12, 7, 6, 7, 7, 9, 5, 4, 12, 10, 2, 7, 6, 7, 8, 7, 8, 8, 7, 7, 7, 6, 8, 2, 4, 10, 3, 5, 6, 6, 5, 10, 5, 5, 10, 10, 7, 7, 8, 9, 8, 3, 11, 9, 9, 7, 5, 11, 6, 3, 5, 9, 3, 6, 7, 3, 4, 7, 8, 8, 7, 7, 8, 3, 7, 3, 11, 5, 7, 9, 6, 11, 3, 11, 12, 2, 10, 6, 6, 8, 9, 6, 3, 4, 7, 9, 7, 8, 12, 12, 3, 7, 10, 5, 6, 12, 8, 7, 8, 5, 4, 7, 7, 9, 12, 10, 7, 6, 7, 9, 6, 7, 7, 7, 8, 12, 3, 5, 9, 7, 7, 10, 7, 11, 10, 7, 5, 7, 5, 8, 7, 7, 9, 7, 7, 8, 10, 4, 10, 11, 6, 8, 10, 4, 8, 3, 5, 3, 8, 5, 7, 7, 11, 6, 4, 7, 5, 5, 9, 5, 3, 8, 9, 6, 8, 8, 3, 8, 7, 4, 5, 7, 5, 9, 8, 6, 8, 7, 7, 6, 12, 6, 5, 8, 6, 8, 8, 8, 7, 9, 8, 4, 7, 6, 6, 8, 5, 7, 7, 7, 7, 5, 8, 7, 7, 9, 4, 6, 3, 8, 8, 6, 7, 12, 6, 5, 6, 5, 5, 11, 6, 8, 8, 7, 9, 7, 12, 8, 7, 9, 6, 12, 2, 10, 2, 10, 10, 8, 5, 7, 6, 8, 10, 6, 7, 9, 3, 7, 9, 4, 7, 6, 6, 10, 2, 8, 8, 9, 5, 9, 8, 9, 6, 8, 6, 7, 5, 10, 9, 5, 6, 7, 5, 9, 6, 6, 4, 6, 8, 6, 3]
        self.actualTurn = 0
        self.currentTurnPlayer = self.players[0] #self.dummy
        
    def reset(self):
        # self.ctr = controller.ActionController()
        self.dummy = Player.Player(0, self, Strategy)
        self.dummy.victoryPoints = 4
        # self.nplayers = num_players
        # for p in self.players:
        #     p.reset()
        self.ctr.reset()
        self.largestArmyPlayer = self.dummy
        self.longestStreetOwner = self.dummy
        self.longestStreetLength = 4
        self.tmpVisitedEdges = []
        # self.dices = [self.rollDice() for _ in range(1200)]
        self.dices = [4, 9, 7, 8, 6, 9, 6, 3, 6, 5, 6, 6, 10, 7, 7, 12, 11, 2, 5, 7, 5, 8, 9, 8, 9, 6, 9, 4, 4, 5, 10, 4, 8, 7, 7, 3, 6, 2, 10, 5, 6, 9, 8, 10, 7, 5, 6, 5, 7, 3, 8, 7, 8, 4, 8, 6, 2, 10, 12, 8, 7, 10, 5, 7, 9, 10, 7, 8, 8, 11, 6, 5, 7, 6, 8, 10, 9, 9, 7, 12, 5, 9, 7, 7, 7, 7, 10, 4, 11, 6, 5, 5, 9, 11, 11, 6, 4, 6, 10, 12, 8, 8, 6, 4, 3, 6, 7, 7, 4, 6, 7, 5, 9, 11, 4, 11, 6, 6, 8, 4, 4, 6, 8, 5, 10, 7, 7, 5, 6, 5, 2, 10, 11, 5, 9, 6, 7, 7, 5, 9, 7, 11, 7, 4, 9, 7, 7, 5, 10, 9, 4, 8, 9, 8, 10, 2, 10, 2, 6, 3, 10, 4, 4, 9, 8, 10, 8, 6, 6, 3, 10, 7, 4, 8, 7, 7, 3, 10, 7, 10, 6, 8, 8, 11, 12, 10, 9, 8, 6, 9, 6, 5, 7, 5, 8, 10, 8, 7, 6, 9, 5, 6, 7, 5, 5, 9, 6, 6, 3, 4, 11, 8, 6, 2, 9, 6, 7, 8, 8, 6, 11, 8, 8, 7, 11, 8, 9, 4, 4, 6, 4, 11, 7, 8, 3, 3, 8, 7, 3, 6, 4, 8, 12, 6, 11, 7, 4, 5, 8, 5, 6, 3, 8, 4, 6, 7, 8, 9, 9, 10, 9, 9, 2, 9, 7, 8, 10, 10, 4, 7, 9, 3, 11, 6, 9, 6, 6, 12, 11, 8, 11, 5, 7, 8, 10, 11, 8, 7, 2, 6, 8, 6, 5, 7, 2, 11, 8, 10, 9, 5, 7, 3, 6, 7, 6, 6, 7, 9, 3, 8, 3, 5, 9, 6, 7, 9, 8, 5, 4, 8, 8, 10, 8, 10, 7, 7, 9, 8, 9, 6, 5, 8, 4, 5, 5, 5, 7, 6, 7, 6, 10, 9, 10, 4, 10, 6, 7, 5, 4, 7, 5, 9, 6, 4, 8, 6, 5, 8, 6, 10, 8, 3, 6, 10, 8, 6, 8, 11, 5, 4, 9, 11, 6, 10, 6, 4, 8, 5, 8, 6, 3, 7, 5, 10, 7, 6, 8, 9, 7, 11, 8, 8, 4, 7, 10, 10, 4, 6, 8, 6, 10, 12, 5, 2, 5, 4, 6, 6, 10, 11, 3, 7, 8, 5, 7, 6, 10, 4, 8, 7, 8, 6, 5, 2, 5, 11, 5, 2, 11, 6, 8, 10, 3, 7, 8, 6, 3, 7, 8, 6, 10, 3, 4, 10, 11, 8, 4, 7, 4, 9, 3, 11, 2, 6, 8, 2, 8, 7, 8, 12, 2, 7, 4, 4, 12, 9, 5, 7, 9, 4, 7, 8, 5, 4, 11, 8, 9, 7, 5, 8, 4, 8, 11, 6, 10, 9, 8, 7, 5, 6, 5, 8, 8, 10, 7, 8, 7, 8, 5, 4, 12, 8, 3, 7, 4, 11, 7, 5, 9, 8, 5, 9, 6, 8, 7, 8, 7, 7, 7, 4, 8, 9, 6, 7, 7, 2, 8, 6, 8, 7, 3, 6, 10, 5, 9, 6, 8, 6, 8, 5, 5, 4, 4, 7, 6, 7, 4, 7, 9, 6, 3, 8, 9, 10, 8, 7, 8, 11, 5, 2, 6, 6, 10, 8, 7, 8, 3, 6, 4, 12, 3, 9, 9, 3, 7, 11, 10, 5, 9, 4, 4, 7, 3, 6, 11, 5, 11, 6, 8, 4, 4, 10, 9, 11, 5, 7, 3, 7, 8, 7, 7, 12, 7, 7, 5, 9, 9, 3, 4, 3, 6, 8, 5, 9, 7, 5, 6, 7, 8, 5, 11, 8, 7, 8, 7, 5, 6, 5, 3, 2, 8, 7, 7, 8, 9, 11, 9, 7, 3, 8, 10, 2, 7, 10, 2, 8, 8, 6, 8, 10, 4, 7, 3, 4, 7, 6, 8, 2, 4, 2, 6, 5, 6, 9, 8, 9, 7, 5, 2, 10, 7, 7, 12, 10, 6, 4, 5, 6, 7, 9, 3, 9, 9, 6, 2, 8, 7, 6, 6, 3, 8, 6, 7, 8, 9, 5, 5, 5, 7, 6, 6, 5, 8, 9, 9, 3, 7, 9, 6, 7, 6, 3, 10, 3, 8, 10, 6, 8, 6, 5, 3, 6, 6, 2, 7, 4, 5, 9, 10, 4, 12, 6, 4, 5, 5, 7, 7, 5, 2, 6, 4, 5, 3, 11, 7, 8, 4, 8, 6, 2, 3, 8, 12, 8, 7, 5, 8, 11, 8, 8, 6, 7, 7, 10, 12, 9, 8, 7, 8, 8, 3, 4, 4, 5, 5, 8, 10, 6, 5, 8, 9, 9, 8, 6, 7, 4, 11, 6, 10, 9, 5, 7, 6, 9, 7, 9, 5, 3, 11, 7, 10, 5, 8, 4, 4, 10, 5, 6, 7, 4, 6, 6, 4, 10, 11, 4, 5, 5, 5, 7, 9, 3, 10, 10, 6, 12, 2, 9, 4, 6, 9, 5, 7, 5, 3, 6, 7, 6, 11, 4, 7, 5, 2, 10, 6, 9, 12, 8, 8, 7, 11, 5, 8, 7, 8, 11, 9, 12, 7, 6, 4, 7, 7, 2, 5, 5, 9, 9, 8, 5, 8, 9, 6, 9, 10, 5, 4, 5, 4, 5, 6, 5, 6, 3, 8, 6, 7, 6, 6, 4, 8, 2, 7, 4, 5, 9, 5, 7, 11, 6, 4, 11, 4, 7, 9, 4, 9, 8, 6, 8, 5, 7, 6, 7, 9, 9, 10, 12, 7, 12, 7, 6, 7, 7, 9, 5, 4, 12, 10, 2, 7, 6, 7, 8, 7, 8, 8, 7, 7, 7, 6, 8, 2, 4, 10, 3, 5, 6, 6, 5, 10, 5, 5, 10, 10, 7, 7, 8, 9, 8, 3, 11, 9, 9, 7, 5, 11, 6, 3, 5, 9, 3, 6, 7, 3, 4, 7, 8, 8, 7, 7, 8, 3, 7, 3, 11, 5, 7, 9, 6, 11, 3, 11, 12, 2, 10, 6, 6, 8, 9, 6, 3, 4, 7, 9, 7, 8, 12, 12, 3, 7, 10, 5, 6, 12, 8, 7, 8, 5, 4, 7, 7, 9, 12, 10, 7, 6, 7, 9, 6, 7, 7, 7, 8, 12, 3, 5, 9, 7, 7, 10, 7, 11, 10, 7, 5, 7, 5, 8, 7, 7, 9, 7, 7, 8, 10, 4, 10, 11, 6, 8, 10, 4, 8, 3, 5, 3, 8, 5, 7, 7, 11, 6, 4, 7, 5, 5, 9, 5, 3, 8, 9, 6, 8, 8, 3, 8, 7, 4, 5, 7, 5, 9, 8, 6, 8, 7, 7, 6, 12, 6, 5, 8, 6, 8, 8, 8, 7, 9, 8, 4, 7, 6, 6, 8, 5, 7, 7, 7, 7, 5, 8, 7, 7, 9, 4, 6, 3, 8, 8, 6, 7, 12, 6, 5, 6, 5, 5, 11, 6, 8, 8, 7, 9, 7, 12, 8, 7, 9, 6, 12, 2, 10, 2, 10, 10, 8, 5, 7, 6, 8, 10, 6, 7, 9, 3, 7, 9, 4, 7, 6, 6, 10, 2, 8, 8, 9, 5, 9, 8, 9, 6, 8, 6, 7, 5, 10, 9, 5, 6, 7, 5, 9, 6, 6, 4, 6, 8, 6, 3]
        # print(self.dices)
        self.actualTurn = 0
        self.currentTurnPlayer = None

    def rollDice(self): 
        return random.randint(1,6) + random.randint(1,6)    

    # def checkWon(self, player):
    #     if(player.victoryPoints >= 10):
    #         return True
    #     return False

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
    
    # def longestStreetPlayer(self):
    #     maxLength = max([self.longest(self.longestStreetOwner), 4])
    #     if(maxLength > 4):
    #         belonger = self.longestStreetOwner
    #     else:
    #         belonger = self.dummy
    #     for p in self.players:
    #         if(p.id != self.longestStreetOwner.id):
    #             actual = self.longest(p)
    #             if(maxLength < actual):
    #                 maxLength = actual
    #                 belonger = p
    #     return belonger, maxLength

    # def longest(self, player):
    #     max = 0
    #     visited = set()
    #     for tail in self.findLeaves(player):
    #         #self.order = 0
    #         length, tmpVisited = self.explorePlace(player, tail, [])
    #         visited.update(tmpVisited)
    #         if max<length:
    #             max = length

    #     for edge in player.ownedStreets:
    #         if edge not in visited:
    #             p1, p2 = edge
    #             length1, tmpVisited1 = self.explorePlace(player, p1, [])
    #             length2, tmpVisited2 = self.explorePlace(player, p2, []) # for fun
    #             visited.update(tmpVisited1)
    #             visited.update(tmpVisited2)
    #             if max < length1:
    #                 max = length1
    #             elif max < length2:
    #                 max = length2
    #     return max - 1 
        
    # def explorePlace(self, player, place, visited):
    #     max = 0
    #     tmpVisited = list.copy(visited)
    #     outVisited = list.copy(visited)
    #     for adjPlace in self.connectedPlacesToPlace(player, place):
    #         edge = tuple(sorted([place, adjPlace]))
    #         if edge not in tmpVisited:
    #             tmpVisited.append(edge)
    #             length, v = self.explorePlace(player, adjPlace, tmpVisited)
    #             outVisited.extend(v)
    #             if(max<length):
    #                 max = length
    #     return max + 1, outVisited

    # def findLeaves(self, player):
    #     toRet = set()
    #     for edge in player.ownedStreets:
    #         p1, p2 = edge
    #         if self.isLeaf(player, p1):
    #             toRet.add(p1)
    #         if self.isLeaf(player, p2):
    #             toRet.add(p2)
    #     return toRet

    # def connectedPlacesToPlace(self, player, place):
    #     toRet = []
    #     if Board.Board().places[place].owner in (0, player.id):
    #         for adjPlace in Board.Board().graph.listOfAdj[place]:
    #             edge = tuple(sorted([place, adjPlace]))
    #             if(Board.Board().edges[edge] == player.id):
    #                 toRet.append(adjPlace)
    #     return toRet   

    # def isLeaf(self, player, place): #  Per il Jack del futuro: non è errato, richiede un po' di ragionamento
    #     return len(self.connectedPlacesToPlace(player, place))==1 or len(self.connectedPlacesToPlace(player, place))==3 # se == 3 allora si trova in un punto ciclico.
    
    def longestStreetPlayer(self):
        maxLength = max([self.longestStreetLength, 4])
        belonger = self.longestStreetOwner
        for p in self.players:
            if(p.longestStreet() > maxLength):
                belonger = p
                maxLength = p.longestStreet()
                self.longestStreetLength = maxLength
        return belonger, maxLength

    def getTotalState(self, player):
        boardState = Board.Board().boardStateTensor(player)
        playerState = player.globalStateTensor()
        # print(boardState.size())
        # print(playerState.size())
        state = torch.cat((boardState, playerState), dim=0).unsqueeze(0)
        # print(state.size())
        return state