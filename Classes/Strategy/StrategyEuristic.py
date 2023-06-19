import random
from Classes import Bank, Board
from Classes.MoveTypes import ForcedMoveTypes, InitialMoveTypes, TurnMoveTypes
from Classes.staticUtilities import availableResourcesForCity, availableResourcesForColony, availableResourcesForDevCard, availableResourcesForStreet
from Command import commands


class StrategyEuristic:
    def __init__(self):
        ...
        pass 

    def name(self):
        return "Not assigned"

    def bestAction(self, player):
        ...
        pass
            
    def chooseParameters(self, action, player): # il vecchio evaluate
        if(action == commands.PlaceFreeStreetCommand):
            # print("Placing free street")
            return commands.PlaceFreeStreetCommand, self.euristicPlaceStreet(player), None
        
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
            return  commands.PlaceStreetCommand, self.euristicPlaceStreet(player), None
        
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
        ownedColonies = player.ownedColonies # to upgrade in city
        max = 0
        choosenColony = -1
        for colony in ownedColonies:
            if(self.placeValue(Board.Board().places[colony]) > max): # da verificare
                max = self.placeValue(Board.Board().places[colony])
                choosenColony = colony
        return Board.Board().places[choosenColony]
    
    def euristicPlaceColony(self, player):
        possibleColonies = player.calculatePossibleColonies()
        # print("Riga 167, RLS: ", possibleColonies)
        if(len(possibleColonies) == 0):
            print("FATAL ERROR.")
        max = 0
        # choosenColony = -1
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
            resourceCopy[trade[1]] -= Bank.Bank().resourceToAsk(player, trade[1])
            if((len(player.calculatePossibleCities()) > 0 and availableResourcesForCity(resourceCopy)) or (len(player.calculatePossibleColonies()) > 0 and availableResourcesForColony(resourceCopy))):
                return trade
            resourceCopy = player.resources.copy()
            
        for trade in trades:
            resourceCopy[trade[0]] += 1
            resourceCopy[trade[1]] -= Bank.Bank().resourceToAsk(player, trade[1])         
            if(len(player.calculatePossibleStreets()) > 0 and availableResourcesForStreet(resourceCopy)):
                return trade
            resourceCopy = player.resources.copy()  

        for trade in trades:
            resourceCopy[trade[0]] += 1
            resourceCopy[trade[1]] -= Bank.Bank().resourceToAsk(player, trade[1])
            if(availableResourcesForDevCard(resourceCopy)):
                return trade
            resourceCopy = player.resources.copy()
            
        for trade in trades:
            resourceCopy[trade[0]] += 1
            resourceCopy[trade[1]] -= Bank.Bank().resourceToAsk(player, trade[1])
            if(sum(player.resources.values()) >= 7):
                return trade
            resourceCopy = player.resources.copy()
        
        return random.choice(trades)

    def euristicDiscardResource(self, player):
        max = 0
        resToDiscard = None
        for res in player.resources.keys():
            if(player.resources[res] > max):
                resToDiscard = res
                max = player.resources[res]
        return resToDiscard
        
    def euristicPlaceRobber(self, player):
        actualDistanceFromEight = 12
        for tile in Board.Board().tiles:
            blockable = False
            isMyTile = False
            for place in tile.associatedPlaces:
                if(Board.Board().places[place].owner != player.id): # prima c'era place.owner qua
                    blockable = True
                if(Board.Board().places[place].owner == player.id): # prima c'era place.owner qua
                    isMyTile = True
            if(blockable and not isMyTile):
                if(actualDistanceFromEight > abs(tile.number - 8)):
                    actualDistanceFromEight = abs(tile.number - 8)
                    bestTile = tile
        return bestTile.identificator
    
    def euristicPlaceKnight(self, player):
        actualDistanceFromEight = 12
        for tile in Board.Board().tiles:
            blockable = False
            isMyTile = False
            for place in tile.associatedPlaces:
                if(Board.Board().places[place].owner != player.id):
                    blockable = True
                if(Board.Board().places[place].owner == player.id):
                    isMyTile = True
            if(blockable and not isMyTile):
                if(actualDistanceFromEight > abs(tile.number - 8)):
                    actualDistanceFromEight = abs(tile.number - 8)
                    bestTile = tile
        return bestTile.identificator

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
    
    def euristicPlaceFreeStreet(self, player):
        availableStreets = player.calculatePossibleStreets()
        if(len(availableStreets) != 0):
            return random.choice(availableStreets) # per ora random
        return None
    
    def euristicMonopoly(self, player):
        min = 50
        toTake = ""
        for res in Bank.Bank().resources.keys():
            if Bank.Bank().resources[res] < min:
                toTake = res
                min = Bank.Bank().resources[res]
        return toTake
    
    def euristicRoadBuildingCard(self, player):
        availableStreets = player.calculatePossibleStreets()
        if len(availableStreets) < 2:
            return availableStreets[0], None
        edge1 = random.choice(availableStreets)
        edge2 = random.choice(availableStreets)
        while edge2 == edge1:
            edge2 = random.choice(availableStreets)
        return edge1, edge2
    
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
    