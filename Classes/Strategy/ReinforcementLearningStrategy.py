from Classes.MoveTypes import ForcedMoveTypes, InitialMoveTypes, TurnMoveTypes
from Command import commands, controller
from Classes.Strategy.Strategy import Strategy
from RL.DQN import DQNagent

class ReinforcementLearningStrategy(Strategy):
    def __init__(self): # diventerà un singleton
        # self, nInputs, nOutputs, criterion, device
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.macroDQN = DQNagent(nInputs, nOutputs, criterion) # macro rete decisionale

        self.macroDQN = DQNagent(x, y) # macro rete decisionale

    def name(self):
        return "RL"

    def bestAction(self, player):
        if(player.game.actualTurn<player.game.nplayers):
            bestAction = -1 #[commands.FirstChoisecommand]
        elif(player.game.actualTurn<player.game.nplayers*2):
            bestAction = -2 #[commands.SecondChoisecommand]
        else: # ...
            state = player.game.getState(player)
            outputs = self.macroDQN.allOutputs(state) # feed forward
            availableTurnActionsId = player.availableTurnActionsId(player.turnCardUsed)
            bestAction = max(outputs[i] for i in availableTurnActionsId)
        return self.euristicOutput(player, bestAction) # action, thingNeeded
        
    def chooseParameters(self, action, player):
        if(action == commands.PlaceFreeStreetCommand):
            return self.euristicOutput(player, ForcedMoveTypes.PlaceFreeStreet)  
        
        if(action == commands.UseRobberCommand): # Yes they are the same method, but must be differentiated becouse of the count of knights.
            return self.euristicOutput(player, ForcedMoveTypes.UseRobber)  

        if(action == commands.DiscardResourceCommand):
            return self.euristicOutput(player, ForcedMoveTypes.DiscardResource)
        
        if(action == commands.FirstChoiseCommand):
            return self.euristicOutput(player, InitialMoveTypes.InitialFirstChoice)   ####

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
            return self.euristicDiscardResource(player)
        
        if(idAction == ForcedMoveTypes.UseRobber):
            return self.euristicRobber(player)
        
        elif(idAction == InitialMoveTypes.InitialFirstChoice):
            return self.euristicInitialFirstMove(player)
        
        elif(idAction == InitialMoveTypes.InitialSecondChoice):
            return self.euristicInitialSecondMove(player)
        
        elif(idAction == TurnMoveTypes.PassTurn):
            return  None
        
        elif(idAction == TurnMoveTypes.BuyDevCard):
            return  None
        
        elif(idAction == TurnMoveTypes.PlaceStreet):
            return  self.euristicPlaceStreet(player)
        
        elif(idAction == TurnMoveTypes.PlaceColony):
            return  self.euristicPlaceColony(player)
        
        elif(idAction == TurnMoveTypes.PlaceCity):
            return  self.euristicPlaceCity(player)
        
        elif(idAction == TurnMoveTypes.TradeBank):
            return  self.euristicTradeBank(player)
        
        elif(idAction == TurnMoveTypes.UseKnight):
            return  self.euristicKnight(player)
        
        elif(idAction == TurnMoveTypes.UseMonopolyCard):
            return  self.euristicMonopoly(player)
        
        elif(idAction == TurnMoveTypes.UseRoadBuildingCard):
            return  self.euristicRoadBuildingCard(player)
        
        elif(idAction == TurnMoveTypes.UseYearOfPlentyCard):
            return  self.euristicYearOfPlenty(player)
        
    def euristicDiscardResource(self, player):
        resToDiscard = max(player.resources, key = player.resources.get) 
        if(len(resToDiscard) > 1):
            resToDiscard = resToDiscard[0]
        return commands.DiscardResourceCommand, resToDiscard
        
    def euristicPlaceRobber(self, player):
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
        return commands.UseRobberCommand, bestTile
    
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
    
    def diceEvaluationFunction(self, diceValue):
        return 1 / (1 + abs(diceValue - 8))
    
    def placeValue(self, place):
        value = 0
        for tile in place.touchedTiles():
            value += self.resValue(tile.resource) * self.diceEvaluationFunction(tile.number)
        return value
        
    def euristicInitialFirstMove(self, player):
        availablePlaces = player.calculatePossibleInitialColony()

        max = 0
        choosenPlace = -1

        for place in availablePlaces:
            if(self.placeValue(place) > max):
                choosenPlace = place

        return action, place
    
    def euristicInitialSecondMove(self, player):
        ...
        return action, thingNeeded 
        
    def euristicPlaceCity(self, player):
        ...
        return action, thingNeeded, None
    
    def euristicPlaceColony(self, player):
        ...
        return action, thingNeeded, None

    def euristicTradeBank(self, player):
        ...
        return action, thingNeeded, None
    
    def euristicRobber(self, player):
        ...
        return action, thingNeeded, None

    def euristicPLayCard(self, player):
        ...
        return action, thingNeeded, None

    def euristicPlaceStreet(self, player):
        ...
        return action, thingNeeded, None
    
    def euristicKnight(self, player):
        ...
        return action, thingNeeded, None
    
    def euristicMonopoly(self, player):
        ...
        return action, thingNeeded, None
    
    def euristicRoadBuildingCard(self, player):
        ...
        return action, thingNeeded, None
    
    def euristicYearOfPlenty(self, player):
        ...
        return action, thingNeeded, None

    