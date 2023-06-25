from Classes import Bank, Board
from Classes.MoveTypes import *
from Classes.Strategy.StrategyRandom import StrategyRandom
from Classes.staticUtilities import *
from Command import commands, controller
import random

class RandomPlayer(StrategyRandom):
    def name(self):
        return "RAN"

    def bestAction(self, player):  #, previousReward):
        if(player.game.actualTurn<player.game.nplayers):
            return self.chooseParameters(commands.PlaceInitialColonyCommand, player)
        elif(player.game.actualTurn<player.game.nplayers*2):
            return self.chooseParameters(commands.PlaceSecondColonyCommand, player)
        else:
            idActions = player.availableTurnActionsId()
            if(len(idActions) == 1 and idActions[0] == 0):
                return commands.PassTurnCommand, None, True
            bestMove = random.choice(idActions)
        return self.chooseParameters(idToCommand(bestMove), player) 
        