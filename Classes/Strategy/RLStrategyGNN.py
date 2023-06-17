from Classes import Bank, Board
from Classes.MoveTypes import *
from Classes.Strategy.StrategyRLEuristic import StrategyRLEuristic
from Classes.staticUtilities import *
from Command import commands, controller
from RL.DQGNN import DQGNNagent
import random

class ReinforcementLearningStrategyGnn(StrategyRLEuristic):
    def __init__(self): # diventerà un singleton
        print("RL STRATEGY CONSTRUCTOR")
        # self, nInputs, nOutputs, criterion, device
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.macroDQN = DQNagent(nInputs, nOutputs, criterion) # macro rete decisionale

        self.macroDQN = DQGNNagent(11, 10) # macro rete decisionale
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
            return self.chooseParameters(commands.FirstChoiseCommand, player)
        elif(player.game.actualTurn<player.game.nplayers*2):
            return self.chooseParameters(commands.SecondChoiseCommand, player)
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
        