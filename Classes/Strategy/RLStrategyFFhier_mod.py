import torch
from Classes import Bank, Board
from Classes.MoveTypes import *
from Classes.Strategy.StrategyEuristic import StrategyEuristic
from Classes.staticUtilities import *
from Command import commands, controller
from RL.DQN import DQNagent
import random

from RL.L2_DQN_mod import L2DQNagent_mod

class ReinforcementLearningStrategyFfHier_mod(StrategyEuristic):
    def __init__(self, eps):
        self.macroDQN = DQNagent(54*11 + 72 + 9, 10, eps) 
        
        self.streetDQN = L2DQNagent_mod("street", 54*11 + 72 + 9, 72, eps)
        self.colonyDQN = L2DQNagent_mod("colonies", 54*11 + 72 + 9, 54, eps)
        self.tradeDQN = L2DQNagent_mod("trades", 54*11 + 72 + 9, 20, eps)

    def name(self):
        return "RL-FF-HIER"
    
    def getEps(self):
        return self.macroDQN.EPS
    
    def epsDecay(self):
        self.macroDQN.epsDecay()
        self.streetDQN.epsDecay()
        self.colonyDQN.epsDecay()
        self.tradeDQN.epsDecay()

    def bestAction(self, player):  #, previousReward):
        if(player.game.actualTurn<player.game.nplayers):
            return self.chooseParameters(commands.PlaceInitialColonyCommand, player)
        elif(player.game.actualTurn<player.game.nplayers*2):
            return self.chooseParameters(commands.PlaceSecondColonyCommand, player)
        else:
            # graph = Board.Board().boardStateGraph(player)
            boardFeatures = Board.Board().boardStateTensor(player).unsqueeze(dim=0)
            glob = player.globalStateTensor()
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
    
    def chooseParameters(self, action, player): # il vecchio evaluate
        if(action == commands.PlaceFreeStreetCommand):
            # print("Placing free street")
            return commands.PlaceFreeStreetCommand, self.DQNFFPlaceStreet(player), None
        
        elif(action == commands.UseRobberCommand): # Yes they are the same method, but must be differentiated becouse of the count of knights.
            # print("Using robber")
            return commands.UseRobberCommand, self.euristicPlaceRobber(player)

        elif(action == commands.DiscardResourceCommand):
            # print("Discarding resource")
            return commands.DiscardResourceCommand, self.euristicDiscardResource(player)
        
        elif(action == commands.PlaceInitialColonyCommand):
            return commands.PlaceInitialColonyCommand, self.DQNFFPlaceInitialColony(player), None

        elif(action == commands.PlaceInitialStreetCommand):
            # print("Initial STREET Choice")
            return commands.PlaceInitialStreetCommand, self.DQNFFPlaceInitialStreet(player)

        elif(action == commands.PlaceSecondColonyCommand):
            return commands.PlaceSecondColonyCommand, self.DQNFFPlaceInitialColony(player), None
        
        elif(action == commands.PassTurnCommand):
            # print("Pass turn")
            return commands.PassTurnCommand, None, None
        
        elif(action == commands.BuyDevCardCommand):
            # print("Buying dev card")
            return  commands.BuyDevCardCommand, None, None
    
        elif(action == commands.PlaceStreetCommand):
            # print("Placing street")
            return  commands.PlaceStreetCommand, self.DQNFFPlaceStreet(player), None
        
        elif(action == commands.PlaceColonyCommand):
            # print("Place colony")
            return  commands.PlaceColonyCommand, self.DQNFFPlaceColony(player), None

        elif(action == commands.PlaceCityCommand):
            # print("Placing city")
            return  commands.PlaceCityCommand, self.euristicPlaceCity(player), None

        elif(action == commands.TradeBankCommand):
            # print("Trade bank")
            return  commands.TradeBankCommand, self.DQNFFTradeBank(player), None    

        elif(action == commands.UseKnightCommand):
            # print("Use knight card")
            return  commands.UseKnightCommand, self.euristicPlaceKnight(player), None

        elif(action == commands.UseMonopolyCardCommand):
            # print("Use monopoly card")
            return  commands.UseMonopolyCardCommand, self.euristicMonopoly(player), None
        
        elif(action == commands.UseRoadBuildingCardCommand):
            # print("Use road building card")
            return  commands.UseRoadBuildingCardCommand, self.DQNFFPlaceStreet(player), None
        
        elif(action == commands.UseYearOfPlentyCardCommand):
            # print("Use year of plenty card")
            return  commands.UseYearOfPlentyCardCommand, self.euristicYearOfPlenty(player), None
        else:
            print("Non existing move selected.")
        
    def DQNFFPlaceStreet(self, player):
        availableStreetsId = [list(Board.Board().edges.keys()).index(edge) for edge in player.calculatePossibleStreets()]
        state = player.game.getTotalState(player)
        bestStreet = self.streetDQN.step(state, availableStreetsId, self.macroDQN)
        return list(Board.Board().edges.keys())[bestStreet]
    
    def DQNFFPlaceInitialStreet(self, player):
        availableStreetsId = [list(Board.Board().edges.keys()).index(edge) for edge in player.calculatePossibleInitialStreets()]

        state = player.game.getTotalState(player)

        bestStreet = self.streetDQN.step(state, availableStreetsId, self.macroDQN)
        return list(Board.Board().edges.keys())[bestStreet]
    
    def DQNFFPlaceColony(self, player):
        possibleColoniesId = [Board.Board().places.index(place) for place in player.calculatePossibleColonies()]
        state = player.game.getTotalState(player)
        choosenColony = self.colonyDQN.step(state, possibleColoniesId, self.macroDQN)
        return Board.Board().places[choosenColony]
    
    def DQNFFPlaceInitialColony(self, player):
        possibleColoniesId = [Board.Board().places.index(place) for place in player.calculatePossibleInitialColonies()]
        state = player.game.getTotalState(player)
        choosenColony = self.colonyDQN.step(state, possibleColoniesId, self.macroDQN)
        return Board.Board().places[choosenColony]
    
    def DQNFFTradeBank(self, player):
        trades = player.calculatePossibleTrades()
        tradesIds = []
        for trade in trades:
            tradesIds.append(tradesToId(trade))
        state = player.game.getTotalState(player)
        choosenTrade = self.tradeDQN.step(state, tradesIds, self.macroDQN)
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


