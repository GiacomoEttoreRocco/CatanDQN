import torch
from Classes import Bank, Board
from Classes.MoveTypes import *
from Classes.Strategy.StrategyEuristic import StrategyEuristic
from Classes.staticUtilities import *
from Command import commands, controller
from RL.DQN import DQNagent
import random

class ReinforcementLearningStrategyFf(StrategyEuristic):
    def __init__(self): # diventerà un singleton
        # print("RL STRATEGY CONSTRUCTOR")

        self.macroDQN = DQNagent(54*11 + 72 + 9, 10) 

    def name(self):
        return "RL-FF"
    
    def getEps(self):
        return self.macroDQN.EPS
    
    def epsDecay(self):
        self.macroDQN.epsDecay()
        # self.eps = self.macroDQN.EPS

    def bestAction(self, player):  #, previousReward):
        if(player.game.actualTurn<player.game.nplayers):
            return self.chooseParameters(commands.FirstChoiseCommand, player)
        elif(player.game.actualTurn<player.game.nplayers*2):
            return self.chooseParameters(commands.SecondChoiseCommand, player)
        else:
            # graph = Board.Board().boardStateGraph(player)
            boardFeatures = Board.Board().boardStateTensor(player).unsqueeze(dim=0)
            glob = player.globalFeaturesToTensor()
            # print("Riga 38, RLSFF: ", boardFeatures.unsqueeze(0))
            # print("Riga 38, RLSFF: ", glob)
            # print("Dimensioni di boardFeatures:", boardFeatures.size())
            # print("Dimensioni di glob:", glob.size())
            state = torch.cat([boardFeatures, glob], dim=1)
            # RICORDATI CHE VANNO GESTITE LE FORCED MOVES, in futuro.

            idActions = player.availableTurnActionsId()
            if(len(idActions) == 1 and idActions[0] == 0):
                return commands.PassTurnCommand, None, True
            # bestMove = self.macroDQN.step(graph, glob, player.availableTurnActionsId()) 
            bestMove = self.macroDQN.step(state, player.availableTurnActionsId()) 

            # print("Best move RL, riga 36 RLStrategy: index: ", bestMove, "Move: ", idToCommand(bestMove))
            # print(" -> ", bestMove[0][0], " move: ", idToCommand(bestMove[0][0]))
        return self.chooseParameters(idToCommand(bestMove), player) # bestAction, thingsNeeded, onlyPassTurn
