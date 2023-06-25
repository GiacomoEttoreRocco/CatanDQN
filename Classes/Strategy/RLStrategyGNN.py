from Classes import Bank, Board
from Classes.MoveTypes import *
from Classes.Strategy.StrategyEuristic import StrategyEuristic
from Classes.staticUtilities import *
from Command import commands, controller
from RL.DQGNN import DQGNNagent
import random

class ReinforcementLearningStrategyGnn(StrategyEuristic):
    def __init__(self, eps): # diventerà un singleton
        # print("RL STRATEGY CONSTRUCTOR")
        self.macroDQN = DQGNNagent(11, 10, eps) # macro rete decisionale
        # self.eps = self.macroDQN.EPS

    def name(self):
        return "RL-GNN"
    
    def getEps(self):
        return self.macroDQN.EPS
    
    def epsDecay(self):
        self.macroDQN.epsDecay()
        # self.eps = self.macroDQN.EPS

    def bestAction(self, player):  #, previousReward):
        if(player.game.actualTurn<player.game.nplayers):
            return self.chooseParameters(commands.PlaceInitialColonyCommand, player)
        elif(player.game.actualTurn<player.game.nplayers*2):
            return self.chooseParameters(commands.PlaceSecondColonyCommand, player)
        else:
            graph = Board.Board().boardStateGraph(player)
            glob = player.globalFeaturesToTensor()
            # RICORDATI CHE VANNO GESTITE LE FORCED MOVES, in futuro.
            idActions = player.availableTurnActionsId()
            if(len(idActions) == 1 and idActions[0] == 0):
                return commands.PassTurnCommand, None, True
            bestMove = self.macroDQN.step(graph, glob, player.availableTurnActionsId()) 
            # print("Best move RL, riga 36 RLStrategy: index: ", bestMove, "Move: ", idToCommand(bestMove))
            # print(" -> ", bestMove[0][0], " move: ", idToCommand(bestMove[0][0]))
        return self.chooseParameters(idToCommand(bestMove), player) # bestAction, thingsNeeded, onlyPassTurn
    
    def saveWeights(self, filepath):
        print("Saving weights...")
        self.macroDQN.policy_net.save_weights(filepath)
        print("Successfully saved.")

    def loadWeights(self, filepath):
        print("Starting loading weights...")
        self.macroDQN.policy_net.load_weights(filepath)
        self.macroDQN.target_net.load_weights(filepath)
        print("Successfully loaded.")

        