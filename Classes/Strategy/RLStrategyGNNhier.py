from Classes import Bank, Board
from Classes.MoveTypes import *
from Classes.Strategy.StrategyEuristic import StrategyEuristic
from Classes.staticUtilities import *
from Command import commands, controller
from RL.DQGNN import DQGNNagent
import random

from RL.L2_DQGNN import L2DQGNNagent

class RLStrategyGnnHierarchical(StrategyEuristic):
    def __init__(self): # diventerà un singleton
        self.macroDQN = DQGNNagent(11, 10) # macro rete decisionale

        self.streetDQN = L2DQGNNagent("street", 11, 72)
        self.colonyDQN = L2DQGNNagent("colonies", 11, 54)

        self.tradeDQN = L2DQGNNagent("trades", 11, 20)

        # self.initialColonyDQN = DQGNNagent(11, 54)

    def name(self):
        return "RL-GNN-HIER"
    
    def getEps(self):
        return self.macroDQN.EPS #, self.streetDQN.EPS
    
    def epsDecay(self):
        self.macroDQN.epsDecay()
        self.streetDQN.epsDecay()
        self.colonyDQN.epsDecay()
        self.tradeDQN.epsDecay()
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
            # return commands.FirstChoiseCommand, self.euristicInitialFirstMove(player), None
            return commands.FirstChoiseCommand, self.DQNPlaceInitialColony(player), None

        elif(action == commands.PlaceInitialStreetCommand):
            # print("Initial STREET Choice")
            return commands.PlaceInitialStreetCommand, self.euristicPlaceInitialStreet(player)

        elif(action == commands.SecondChoiseCommand):
            # print("Initial SECOND choice")
            # return commands.SecondChoiseCommand, self.euristicInitialSecondMove(player), None
            return commands.SecondChoiseCommand, self.DQNPlaceInitialColony(player), None
        
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
            return  commands.PlaceColonyCommand, self.DQNPlaceColony(player), None

        elif(action == commands.PlaceCityCommand):
            # print("Placing city")
            return  commands.PlaceCityCommand, self.euristicPlaceCity(player), None

        elif(action == commands.TradeBankCommand):
            # print("Trade bank")
            return  commands.TradeBankCommand, self.DQNTradeBank(player), None    

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
        bestStreet = self.streetDQN.step(graph, glob, availableStreetsId, self.macroDQN)
        # print(bestStreet)
        return list(Board.Board().edges.keys())[bestStreet]
    
    def DQNPlaceColony(self, player):
        # print("Specialized colony placed.")
        possibleColoniesId = [Board.Board().places.index(place) for place in player.calculatePossibleColonies()]

        graph = Board.Board().boardStateGraph(player)
        glob = player.globalFeaturesToTensor()
        choosenColony = self.colonyDQN.step(graph, glob, possibleColoniesId, self.macroDQN)
        # print(choosenColony)
        return Board.Board().places[choosenColony]
    
    def DQNPlaceInitialColony(self, player):
        # print("Specialized initial colony placed.")
        possibleColoniesId = [Board.Board().places.index(place) for place in player.calculatePossibleInitialColonies()]
        graph = Board.Board().boardStateGraph(player)
        glob = player.globalFeaturesToTensor()
        choosenColony = self.colonyDQN.step(graph, glob, possibleColoniesId, self.macroDQN)
        # print(choosenColony)
        return Board.Board().places[choosenColony]
    
    def DQNTradeBank(self, player):
        # print("trade spec")
        trades = player.calculatePossibleTrades()
        tradesIds = []
        for trade in trades:
            tradesIds.append(tradesToId(trade))

        graph = Board.Board().boardStateGraph(player)
        glob = player.globalFeaturesToTensor()

        choosenTrade = self.tradeDQN.step(graph, glob, tradesIds, self.macroDQN)
        return idToTrade(choosenTrade)
    
    def saveWeights(self, filepath):
        print("Saving weights...")
        self.macroDQN.policy_net.save_weights(filepath+".pth")
        self.streetDQN.policy_net.save_weights(filepath+"street.pth")
        self.colonyDQN.policy_net.save_weights(filepath+"colony.pth")
        self.tradeDQN.policy_net.save_weights(filepath+"trade.pth")
        print("Successfully saved.")

    def loadWeights(self, filepath):
        print("Starting loading weights...")
        self.macroDQN.policy_net.load_weights(filepath+".pth")
        self.macroDQN.target_net.load_weights(filepath+".pth")

        self.streetDQN.policy_net.load_weights(filepath+"street.pth")
        self.streetDQN.target_net.load_weights(filepath+"street.pth")

        self.colonyDQN.policy_net.load_weights(filepath+"colony.pth")
        self.colonyDQN.target_net.load_weights(filepath+"colony.pth")

        self.tradeDQN.policy_net.load_weights(filepath+"trade.pth")
        self.tradeDQN.target_net.load_weights(filepath+"trade.pth")
        print("Successfully loaded.")


    

    
    
    
