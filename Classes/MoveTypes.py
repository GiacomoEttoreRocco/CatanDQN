from enum import Enum, auto

from Command import commands

class TurnMoveTypes(Enum):
    PassTurn = 0
    BuyDevCard = 1
    PlaceStreet = 2
    PlaceColony = 3
    PlaceCity = 4
    TradeBank = 5
    UseKnight = 6
    UseMonopolyCard = 7
    UseRoadBuildingCard = 8
    UseYearOfPlentyCard = 9

class ForcedMoveTypes(Enum): # se sono forced non ha senso farle valutare dal DQN globale, solo dal DQN specifico
    PlaceFreeStreet = -6
    UseRobber = -5
    DiscardResource = -4

class InitialMoveTypes(Enum): # idem
    InitialFirstChoice = -3
    InitialStreetChoice = -2
    InitialSecondChoice = -1

def idToCommand(id):
    if id == -6:
        return commands.PlaceFreeStreetCommand
    elif id == -5:
        return commands.UseRobberCommand
    elif id == -4:
        return commands.DiscardResourceCommand
    elif id == -3:
        return commands.FirstChoiseCommand
    elif id == -2:
        return commands.PlaceInitialStreetCommand
    elif id == -1:
        return commands.SecondChoiseCommand
    elif id == 0:
        return commands.PassTurnCommand
    elif id == 1:
        return commands.BuyDevCardCommand
    elif id == 2:
        return commands.PlaceStreetCommand
    elif id == 3:
        return commands.PlaceColonyCommand
    elif id == 4:
        return commands.PlaceCityCommand
    elif id == 5:
        return commands.TradeBankCommand
    elif id == 6:
        return commands.UseKnightCommand
    elif id == 7:
        return commands.UseMonopolyCardCommand
    elif id == 8:
        return commands.UseRoadBuildingCardCommand
    elif id == 9:
        return commands.UseYearOfPlentyCardCommand
