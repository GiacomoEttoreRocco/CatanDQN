from Classes import Bank, Board
from Classes.MoveTypes import *
from Classes.Strategy.StrategyEuristic import StrategyEuristic
from Classes.staticUtilities import *
from Command import commands, controller
from RL.DQGNN import DQGNNagent
import random

class RLStrategyGnnStreet(StrategyEuristic):
    def __init__(self): # diventerà un singleton
        self.macroDQN = DQGNNagent(11, 10) # macro rete decisionale
        self.streetDQN = DQGNNagent(11, 72)

    def name(self):
        return "RL-GNN-STREET"
    
    def getEps(self):
        return self.macroDQN.EPS, self.streetDQN.EPS
    
    def epsDecay(self):
        self.macroDQN.epsDecay()
        self.streetDQN.epsDecay()
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
        return self.chooseParameters(idToCommand(bestMove), player) # bestAction, thingsNeeded, onlyPassTurn
    
    def chooseParameters(self, action, player): # il vecchio evaluate
        if(action == commands.PlaceFreeStreetCommand):
            # print("Placing free street")
            return commands.PlaceFreeStreetCommand, self.DQNPlaceStreet(player), None
        
        elif(action == commands.UseRobberCommand): # Yes they are the same method, but must be differentiated becouse of the count of knights.
            # print("Using robber")
            return commands.UseRobberCommand, self.euristicPlaceRobber(player)

        elif(action == commands.DiscardResourceCommand):
            # print("Discarding resource")
            return commands.DiscardResourceCommand, self.euristicDiscardResource(player)
        
        elif(action == commands.FirstChoiseCommand):
            # print("InitialFIRSTChoice")
            return commands.FirstChoiseCommand, self.euristicInitialFirstMove(player), None
        
        elif(action == commands.PlaceInitialStreetCommand):
            # print("Initial STREET Choice")
            return commands.PlaceInitialStreetCommand, self.euristicPlaceInitialStreet(player)

        elif(action == commands.SecondChoiseCommand):
            # print("Initial SECOND choice")
            return commands.SecondChoiseCommand, self.euristicInitialSecondMove(player), None
        
        elif(action == commands.PassTurnCommand):
            # print("Pass turn")
            return commands.PassTurnCommand, None, None
        
        elif(action == commands.BuyDevCardCommand):
            # print("Buying dev card")
            return  commands.BuyDevCardCommand, None, None
    
        elif(action == commands.PlaceStreetCommand):
            # print("Placing street")
            return  commands.PlaceStreetCommand, self.DQNPlaceStreet(player), None
        
        elif(action == commands.PlaceColonyCommand):
            # print("Place colony")
            return  commands.PlaceColonyCommand, self.euristicPlaceColony(player), None

        elif(action == commands.PlaceCityCommand):
            # print("Placing city")
            return  commands.PlaceCityCommand, self.euristicPlaceCity(player), None

        elif(action == commands.TradeBankCommand):
            # print("Trade bank")
            return  commands.TradeBankCommand, self.euristicTradeBank(player), None    

        elif(action == commands.UseKnightCommand):
            # print("Use knight card")
            return  commands.UseKnightCommand, self.euristicPlaceKnight(player), None

        elif(action == commands.UseMonopolyCardCommand):
            # print("Use monopoly card")
            return  commands.UseMonopolyCardCommand, self.euristicMonopoly(player), None
        
        elif(action == commands.UseRoadBuildingCardCommand):
            # print("Use road building card")
            return  commands.UseRoadBuildingCardCommand, self.euristicRoadBuildingCard(player), None
        
        elif(action == commands.UseYearOfPlentyCardCommand):
            # print("Use year of plenty card")
            return  commands.UseYearOfPlentyCardCommand, self.euristicYearOfPlenty(player), None
        else:
            print("Non existing move selected.")
        
    def DQNPlaceStreet(self, player):
        # print("Specialized street")
        # availableStreets = player.calculatePossibleStreets()
        availableStreetsId = [list(Board.Board().edges.keys()).index(edge) for edge in player.calculatePossibleStreets()]

        graph = Board.Board().boardStateGraph(player)
        glob = player.globalFeaturesToTensor()
        bestStreet = self.streetDQN.step(graph, glob, availableStreetsId)
        # print(bestStreet)
        return list(Board.Board().edges.keys())[bestStreet]

    # def euristicPlaceStreet(self, player):
    #     availableStreets = player.calculatePossibleStreets()
    #     # print(availableStreets)
    #     if(len(availableStreets) != 0):
    #         return random.choice(availableStreets) # per ora random
    #     return None