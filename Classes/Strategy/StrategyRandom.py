import random
from Classes import Bank, Board
from Classes.MoveTypes import ForcedMoveTypes, InitialMoveTypes, TurnMoveTypes
from Classes.staticUtilities import availableResourcesForCity, availableResourcesForColony, availableResourcesForDevCard, availableResourcesForStreet, blockableTile
from Command import commands


class StrategyRandom:
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
            return commands.PlaceFreeStreetCommand, self.randomicPlaceStreet(player), None
        
        elif(action == commands.UseRobberCommand): # Yes they are the same method, but must be differentiated becouse of the count of knights.
            # print("Using robber")
            return commands.UseRobberCommand, self.randomicPlaceRobber(player)

        elif(action == commands.DiscardResourceCommand):
            # print("Discarding resource")
            return commands.DiscardResourceCommand, self.randomicDiscardResource(player)
        
        elif(action == commands.PlaceInitialColonyCommand):
            # print("InitialFIRSTChoice")
            return commands.PlaceInitialColonyCommand, self.randomicInitialFirstMove(player), None
        
        elif(action == commands.PlaceInitialStreetCommand):
            # print("Initial STREET Choice")
            return commands.PlaceInitialStreetCommand, self.randomicPlaceInitialStreet(player)

        elif(action == commands.PlaceSecondColonyCommand):
            # print("Initial SECOND choice")
            return commands.PlaceSecondColonyCommand, self.randomicInitialSecondMove(player), None
        
        elif(action == commands.PassTurnCommand):
            # print("Pass turn")
            return commands.PassTurnCommand, None, None
        
        elif(action == commands.BuyDevCardCommand):
            # print("Buying dev card")
            return  commands.BuyDevCardCommand, None, None
    
        elif(action == commands.PlaceStreetCommand):
            # print("Placing street")
            return  commands.PlaceStreetCommand, self.randomicPlaceStreet(player), None
        
        elif(action == commands.PlaceColonyCommand):
            # print("Place colony")
            return  commands.PlaceColonyCommand, self.randomicPlaceColony(player), None

        elif(action == commands.PlaceCityCommand):
            # print("Placing city")
            return  commands.PlaceCityCommand, self.randomicPlaceCity(player), None

        elif(action == commands.TradeBankCommand):
            # print("Trade bank")
            return  commands.TradeBankCommand, self.randomicTradeBank(player), None    

        elif(action == commands.UseKnightCommand):
            # print("Use knight card")
            return  commands.UseKnightCommand, self.randomicPlaceKnight(player), None

        elif(action == commands.UseMonopolyCardCommand):
            # print("Use monopoly card")
            return  commands.UseMonopolyCardCommand, self.randomicMonopoly(player), None
        
        elif(action == commands.UseRoadBuildingCardCommand):
            # print("Use road building card")
            return  commands.UseRoadBuildingCardCommand, self.randomicRoadBuildingCard(player), None
        
        elif(action == commands.UseYearOfPlentyCardCommand):
            # print("Use year of plenty card")
            return  commands.UseYearOfPlentyCardCommand, self.randomicYearOfPlenty(player), None
        else:
            print("Non existing move selected.")
    
    def resValue(self, resource):
        if(resource == "iron"):
            return 0.5
        elif(resource == "crop"):
            return 0.5
        elif(resource == "wood"):
            return 0.4
        elif(resource == "clay"):
            return 0.4
        elif(resource == "sheep"):
            return 0.3
        else:
            return 0
    
    def diceEvaluationFunction(self, diceValue):
        return 7.0 / (1.0 + abs(diceValue - 7.0))
    
    def placeValue(self, place):
        value = 0
        for tile in place.touchedTiles:
            value += self.resValue(Board.Board().tiles[tile].resource) * self.diceEvaluationFunction(Board.Board().tiles[tile].number)
        return value
        
    def randomicInitialFirstMove(self, player):
        availablePlaces = player.calculatePossibleInitialColonies()
        choosenPlace = random.choice(availablePlaces)
        return choosenPlace
    
    def randomicPlaceInitialStreet(self, player):
        availableStreets = player.calculatePossibleInitialStreets()
        return random.choice(availableStreets)
    
    def randomicInitialSecondMove(self, player):
        availablePlaces = player.calculatePossibleInitialColonies()
        choosenPlace = random.choice(availablePlaces)
        return choosenPlace
        
    def randomicPlaceCity(self, player):
        ownedColonies = player.ownedColonies # to upgrade in city
        choosenPlace = random.choice(ownedColonies)
        return Board.Board().places[choosenPlace]
    
    def randomicPlaceColony(self, player):
        possibleColonies = player.calculatePossibleColonies()
        choosenColony = random.choice(possibleColonies)
        return choosenColony

    def randomicTradeBank(self, player):
        trades = player.calculatePossibleTrades()
        return random.choice(trades)

    def randomicDiscardResource(self, player):
        resToDiscard = None
        for res in player.resources.keys():
            if(player.resources[res] > 0):
                resToDiscard = res
        return resToDiscard
        
    def randomicPlaceRobber(self, player):
        bestTile = random.choice(Board.Board().tiles)
        return bestTile.identificator
    
    def randomicPlaceKnight(self, player):
        bestTile = random.choice(Board.Board().tiles)
        return bestTile.identificator

    def randomicPlayCard(self, player):
        if player.unusedKnights > 0:
            return self.randomicPlayKnight(player)
        if player.monopolyCard > 0:
            return self.randomicMonopoly(player)
        if player.roadBuildingCard > 0:
            return self.randomicRoadBuildingCard(player)
        if player.yearOfPlenty > 0:
            return self.randomicYearOfPlenty(player)

    def randomicPlaceStreet(self, player):
        availableStreets = player.calculatePossibleStreets()
        if(len(availableStreets) != 0):
            return random.choice(availableStreets)
        return None
    
    def randomicPlaceFreeStreet(self, player):
        availableStreets = player.calculatePossibleStreets()
        if(len(availableStreets) != 0):
            return random.choice(availableStreets)
        return None
    
    def randomicMonopoly(self, player):
        min = 50
        toTake = random.choice(list(Bank.Bank().resources.keys()))
        return toTake
    
    def randomicRoadBuildingCard(self, player):
        availableStreets = player.calculatePossibleStreets()
        # if len(availableStreets) < 2:
        #     return availableStreets[0], None
        edge1 = random.choice(availableStreets)
        # edge2 = random.choice(availableStreets)
        # while edge2 == edge1:
            # edge2 = random.choice(availableStreets)
        return edge1#, edge2
    
    def randomicYearOfPlenty(self, player):
        resources = ["iron", "wood", "clay", "crop", "sheep"]
        return random.choice(resources), random.choice(resources)
    