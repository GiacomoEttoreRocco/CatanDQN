from enum import Enum, auto

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
    PlaceFreeStreet = -5
    UseRobber = -4
    DiscardResource = -3

class InitialMoveTypes(Enum): # idem
    InitialFirstChoice = -2
    InitialSecondChoice = -1

