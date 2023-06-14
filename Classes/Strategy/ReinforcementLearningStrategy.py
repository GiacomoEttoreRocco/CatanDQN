from Classes import Bank, Board
from Classes.MoveTypes import ForcedMoveTypes, InitialMoveTypes, TurnMoveTypes
from Classes.staticUtilities import *
from Command import commands, controller
from Classes.Strategy.Strategy import Strategy
from RL.DQGNN import DQGNNagent
import random

class ReinforcementLearningStrategy(Strategy):
    def __init__(self): # diventerà un singleton
        
        # self, nInputs, nOutputs, criterion, device
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.macroDQN = DQNagent(nInputs, nOutputs, criterion) # macro rete decisionale

        self.macroDQN = DQGNNagent(11, 10) # macro rete decisionale

    def name(self):
        return "RL"

    def bestAction(self, player):  #, previousReward):
        if(player.game.actualTurn<player.game.nplayers):
            return self.euristicOutput(player, InitialMoveTypes.InitialFirstChoice)
        elif(player.game.actualTurn<player.game.nplayers*2):
            return self.euristicOutput(player, InitialMoveTypes.InitialSecondChoice)
        else: # ...
            # state = player.game.getTotalState(player)

            graph = Board.Board().boardStateGraph(player)
            glob = player.globalFeaturesToTensor()

            # RICORDATI CHE VANNO GESTITE LE FORCED MOVES, in futuro.
            bestMove = self.macroDQN.step(graph, glob, player.availableTurnActionsId()) 
        return self.euristicOutput(player, bestMove) # action, thingNeeded
        
    def chooseParameters(self, action, player):
        if(action == commands.PlaceFreeStreetCommand):
            return self.euristicOutput(player, ForcedMoveTypes.PlaceFreeStreet)
        
        if(action == commands.UseRobberCommand): # Yes they are the same method, but must be differentiated becouse of the count of knights.
            return self.euristicOutput(player, ForcedMoveTypes.UseRobber)  

        if(action == commands.DiscardResourceCommand):
            return self.euristicOutput(player, ForcedMoveTypes.DiscardResource)
        
        if(action == commands.FirstChoiseCommand):
            return self.euristicOutput(player, InitialMoveTypes.InitialFirstChoice)   ####
        
        if(action == commands.PlaceInitialStreetCommand):
            return 0, self.euristicOutput(player, InitialMoveTypes.InitialStreetChoice)

        if(action == commands.SecondChoiseCommand):
            return self.euristicOutput(player, InitialMoveTypes.InitialSecondChoice)
        
        if(action == commands.PassTurnCommand):
            return self.euristicOutput(player, TurnMoveTypes.PassTurn)
        
        if(action == commands.BuyDevCardCommand):
            return self.euristicOutput(player, TurnMoveTypes.BuyDevCard)
    
        if(action == commands.PlaceStreetCommand):
            return self.euristicOutput(player, TurnMoveTypes.PlaceStreet)
        
        if(action == commands.PlaceColonyCommand):
            return self.euristicOutput(player, TurnMoveTypes.PlaceColony)

        if(action == commands.PlaceCityCommand):
            return self.euristicOutput(player, TurnMoveTypes.PlaceCity)

        if(action == commands.TradeBankCommand):
            return self.euristicOutput(player, TurnMoveTypes.TradeBank)        

        if(action == commands.UseKnightCommand):
            return self.euristicOutput(player, TurnMoveTypes.UseKnight)  

        if(action == commands.UseMonopolyCardCommand):
            return self.euristicOutput(player, TurnMoveTypes.UseMonopolyCard)
        
        if(action == commands.UseRoadBuildingCardCommand):
            return self.euristicOutput(player, TurnMoveTypes.UseRoadBuildingCard)
        
        if(action == commands.UseYearOfPlentyCardCommand):
            return self.euristicOutput(player, TurnMoveTypes.UseYearOfPlentyCard)
    
    def euristicOutput(self, player, idAction):
        if(idAction == ForcedMoveTypes.DiscardResource):
            return commands.DiscardResourceCommand, self.euristicDiscardResource(player), None
        
        if(idAction == ForcedMoveTypes.UseRobber):
            return commands.UseRobberCommand, self.euristicPlaceRobber(player), None
        
        elif(idAction == InitialMoveTypes.InitialFirstChoice):
            # print("INITIAL CHOICE: ", commands.FirstChoiseCommand, self.euristicInitialFirstMove(player), None)
            return commands.FirstChoiseCommand, self.euristicInitialFirstMove(player), None

        elif(idAction == InitialMoveTypes.InitialStreetChoice):
            return self.euristicPlaceInitialStreet(player)
        
        elif(idAction == InitialMoveTypes.InitialSecondChoice):
            return commands.SecondChoiseCommand, self.euristicInitialSecondMove(player), None
        
        elif(idAction == TurnMoveTypes.PassTurn):
            return  commands.PassTurnCommand, None, None
        
        elif(idAction == TurnMoveTypes.BuyDevCard):
            return  commands.BuyDevCardCommand, None, None
        
        elif(idAction == TurnMoveTypes.PlaceStreet):
            return  commands.PlaceStreetCommand, self.euristicPlaceStreet(player), None
        
        elif(idAction == TurnMoveTypes.PlaceColony):
            return  commands.PlaceColonyCommand, self.euristicPlaceColony(player), None
        
        elif(idAction == TurnMoveTypes.PlaceCity):
            return  commands.PlaceCityCommand, self.euristicPlaceCity(player), None
        
        elif(idAction == TurnMoveTypes.TradeBank):
            return  commands.TradeBankCommand, self.euristicTradeBank(player), None
        
        elif(idAction == TurnMoveTypes.UseKnight):
            return  commands.UseKnightCommand, self.euristicKnight(player), None
        
        elif(idAction == TurnMoveTypes.UseMonopolyCard):
            return  commands.UseMonopolyCardCommand, self.euristicMonopoly(player), None
        
        elif(idAction == TurnMoveTypes.UseRoadBuildingCard):
            return  commands.UseRoadBuildingCardCommand, self.euristicRoadBuildingCard(player), None
        
        elif(idAction == TurnMoveTypes.UseYearOfPlentyCard):
            return  commands.UseYearOfPlentyCardCommand, self.euristicYearOfPlenty(player), None
    
    def resValue(self, resource):
        if(resource == "iron"):
            return 5
        elif(resource == "crop"):
            return 5
        elif(resource == "wood"):
            return 4
        elif(resource == "clay"):
            return 4
        elif(resource == "sheep"):
            return 3
        else:
            return 0
    
    def diceEvaluationFunction(self, diceValue):
        return 1 / (1 + abs(diceValue - 8))
    
    def placeValue(self, place):
        value = 0
        for tile in place.touchedTiles:
            value += self.resValue(Board.Board().tiles[tile].resource) * self.diceEvaluationFunction(Board.Board().tiles[tile].number)
        return value
        
    def euristicInitialFirstMove(self, player):
        availablePlaces = player.calculatePossibleInitialColonies()
        max = 0
        choosenPlace = -1
        for place in availablePlaces:
            if(self.placeValue(place) > max):
                max = self.placeValue(place)
                choosenPlace = place
        return choosenPlace
    
    def euristicPlaceInitialStreet(self, player):
        availableStreets = player.calculatePossibleInitialStreets()
        return random.choice(availableStreets)
    
    def euristicInitialSecondMove(self, player):
        availablePlaces = player.calculatePossibleInitialColonies()
        max = 0
        choosenPlace = -1
        for place in availablePlaces:
            if(self.placeValue(place) > max):
                max = self.placeValue(place)
                choosenPlace = place
        return choosenPlace
        
    def euristicPlaceCity(self, player):
        ownedColonies = player.ownedColonies() # to upgrade in city
        max = 0
        choosenColony = -1
        for colony in ownedColonies:
            if(self.placeValue(colony) > max):
                max = self.placeValue(colony)
                choosenColony = colony
        return choosenColony
    
    def euristicPlaceColony(self, player):
        possibleColonies = player.calculatePossibleColonies()
        max = 0
        choosenColony = -1
        for colony in possibleColonies:
            if(self.placeValue(colony) > max):
                max = self.placeValue(colony)
                choosenColony = colony
        return choosenColony

    def euristicTradeBank(self, player):
        trades = player.calculatePossibleTrades()
        resourceCopy = player.resources.copy()
        for trade in trades:
            resourceCopy[trade[0]] += 1
            resourceCopy[trade[1]] -= Bank().resourceToAsk(player, trade[1])
            if((player.calculatePossibleCities() > 0 and availableResourcesForCity(resourceCopy)) or (player.calculatePossibleColonies() and availableResourcesForColony(resourceCopy))):
                return trade
            resourceCopy = player.resources.copy()
            
        for trade in trades:
            resourceCopy[trade[0]] += 1
            resourceCopy[trade[1]] -= Bank().resourceToAsk(player, trade[1])         
            if(player.calculatePossibleStreets() > 0 and availableResourcesForStreet(resourceCopy)):
                return trade
            resourceCopy = player.resources.copy()  

        for trade in trades:
            resourceCopy[trade[0]] += 1
            resourceCopy[trade[1]] -= Bank().resourceToAsk(player, trade[1])
            if(availableResourcesForDevCard(resourceCopy)):
                return trade
            resourceCopy = player.resources.copy()
            
        for trade in trades:
            resourceCopy[trade[0]] += 1
            resourceCopy[trade[1]] -= Bank().resourceToAsk(player, trade[1])
            if(sum(player.resources) >= 7):
                return trade
            resourceCopy = player.resources.copy()

    def euristicDiscardResource(self, player):
        resToDiscard = max(player.resources, key = player.resources.get) 
        if(len(resToDiscard) > 1):
            resToDiscard = resToDiscard[0]
        return resToDiscard
        
    def euristicPlaceRobber(self, player):
        actualDistanceFromEight = 12
        for tile in Board.Board().tiles:
            blockable = False
            isMyTile = False
            for place in tile.associatedPlaces:
                # print(place, "-" , player.id)
                if(Board.Board().places[place].owner != player.id): # prima c'era place.owner qua
                    blockable = True
                if(Board.Board().places[place].owner == player.id): # prima c'era place.owner qua
                    isMyTile = True
                if(blockable and not isMyTile):
                    if(actualDistanceFromEight > abs(tile.number - 8)):
                        actualDistanceFromEight = abs(tile.number - 8)
                        bestTile = tile
        return bestTile

    def euristicPlayCard(self, player):
        if player.unusedKnights > 0:
            return self.euristicPlayKnight(player)
        if player.monopolyCard > 0:
            return self.euristicMonopoly(player)
        if player.roadBuildingCard > 0:
            return self.euristicRoadBuildingCard(player)
        if player.yearOfPlenty > 0:
            return self.euristicYearOfPlenty(player)

    def euristicPlaceStreet(self, player):
        availableStreets = player.calculatePossibleStreets()
        if(len(availableStreets) != 0):
            return random.choice(availableStreets) # per ora random
        return None
    
    def euristicKnight(self, player):
        actualDistanceFromEight = 12
        for tile in player.game.tiles:
            blockable = False
            isMyTile = False
            for place in tile.associatedPlaces:
                if(place.owner != player.id):
                    blockable = True
                if(place.owner == player.id):
                    isMyTile = True
            if(blockable and not isMyTile):
                if(actualDistanceFromEight < abs(tile.number, 8)):
                    bestTile = tile
        return bestTile
    
    def euristicMonopoly(self, player):
        min = 25
        toTake = ""
        for res in Bank().resources:
            if Bank.resources[res] < min:
                toTake = res
                min = Bank.resources[res]
        return toTake
    
    def euristicRoadBuildingCard(self, player):
        return self.euristicPlaceStreet(player)  # si può chiamare questo, perchè nel momento in cui viene gestito il command di road building card, viene chiamato 2 volte il comando place street
    
    def euristicYearOfPlenty(self, player):
        resources = ["iron", "wood", "clay", "crop", "sheep"]
        return random.choice(resources), random.choice(resources)

    def getActionId(self, command):
        if(command == commands.DiscardResourceCommand):
            return ForcedMoveTypes.DiscardResource
        
        elif(command == commands.UseRobberCommand):
            return ForcedMoveTypes.UseRobber
        
        elif(command == commands.FirstChoiseCommand):
            return InitialMoveTypes.InitialFirstChoice

        elif(command == commands.PlaceInitialStreetCommand):
            return InitialMoveTypes.InitialStreetChoice
        
        elif(command == commands.SecondChoiseCommand):
            return InitialMoveTypes.InitialSecondChoice
        
        elif(command == commands.PassTurnCommand):
            return TurnMoveTypes.PassTurn
        
        elif(command == commands.BuyDevCardCommand):
            return TurnMoveTypes.BuyDevCard
        
        elif(command == commands.PlaceStreetCommand):
            return TurnMoveTypes.PlaceStreet
        
        elif(command == commands.PlaceColonyCommand):
            return TurnMoveTypes.PlaceColony
        
        elif(command == commands.PlaceCityCommand):
            return TurnMoveTypes.PlaceCity 
        
        elif(command == commands.TradeBankCommand):
            return TurnMoveTypes.TradeBank 
        
        elif(command == commands.UseKnightCommand):
            return TurnMoveTypes.UseKnight 
        
        elif(command == commands.UseMonopolyCardCommand):
            return TurnMoveTypes.UseMonopolyCard  
        
        elif(command == commands.UseRoadBuildingCardCommand):
            return TurnMoveTypes.UseRoadBuildingCard 
        
        elif(command == commands.UseYearOfPlentyCardCommand):
            return TurnMoveTypes.UseYearOfPlentyCard 
    