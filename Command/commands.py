from dataclasses import dataclass, field
from Command.action import Action as Action
import Classes.Player as Player
import Classes.Game as Game
import Classes.Board as Board
import Classes.CatanGraph as cg
import Classes.Bank as Bank
import random
import numpy as np

@dataclass
class InitialTurnSetupCommand:
    player: Player

    oldUnusedKnights : int = 0
    oldJustBoughtKnights : int = 0
    oldMonopolyCard : int = 0
    oldJustBoughtMonopolyCard : int = 0
    oldRoadBuildingCard : int = 0
    oldJustBoughtRoadBuildingCard : int = 0
    oldYearOfPlentyCard : int = 0
    oldJustBoughtYearOfPlentyCard : int = 0
    oldTurnCardUsed : bool = False

    def execute(self):
        self.oldUnusedKnights = self.player.unusedKnights 
        self.oldJustBoughtKnights = self.player.justBoughtKnights
        self.oldMonopolyCard = self.player.monopolyCard 
        self.oldJustBoughtMonopolyCard = self.player.justBoughtMonopolyCard
        self.oldRoadBuildingCard = self.player.roadBuildingCard
        self.oldJustBoughtRoadBuildingCard = self.player.justBoughtRoadBuildingCard
        self.oldYearOfPlentyCard = self.player.yearOfPlentyCard
        self.oldJustBoughtYearOfPlentyCard = self.player.justBoughtYearOfPlentyCard
        self.oldTurnCardUsed = self.player.turnCardUsed

        self.player.unusedKnights = self.player.unusedKnights + self.player.justBoughtKnights
        self.player.justBoughtKnights = 0
        self.player.monopolyCard += self.player.justBoughtMonopolyCard
        self.player.justBoughtMonopolyCard = 0
        self.player.roadBuildingCard += self.player.justBoughtRoadBuildingCard
        self.player.justBoughtRoadBuildingCard = 0
        self.player.yearOfPlentyCard += self.player.justBoughtYearOfPlentyCard
        self.player.justBoughtYearOfPlentyCard = 0
        self.player.turnCardUsed = False

    def undo(self):
        self.player.unusedKnights = self.oldUnusedKnights
        self.player.justBoughtKnights = self.oldJustBoughtKnights 
        self.player.monopolyCard = self.oldMonopolyCard
        self.player.justBoughtMonopolyCard = self.oldJustBoughtMonopolyCard
        self.player.roadBuildingCard = self.oldRoadBuildingCard
        self.player.justBoughtRoadBuildingCard = self.oldJustBoughtRoadBuildingCard
        self.player.yearOfPlentyCard = self.oldYearOfPlentyCard
        self.player.justBoughtYearOfPlentyCard = self.oldJustBoughtYearOfPlentyCard
        self.player.turnCardUsed = self.oldTurnCardUsed 

    def redo(self):
        self.execute()

    def __repr__(self) -> str:
        return self.__class__.__name__

@dataclass
class SetRobberTile:
    tilePosition: cg.Tile
    previousPosition: cg.Tile = None

    def execute(self):
        self.previousPosition = Board.Board().robberTile
        Board.Board().robberTile = self.tilePosition
    def undo(self):
        Board.Board().robberTile = self.previousPosition
    def redo(self):
        self.execute()
    def __repr__(self) -> str:
        return f'{self.__class__.__name__} tile: {self.tilePosition}'


@dataclass
class UseRobberCommand:
    player: Player
    tilePosition: cg.Tile
    actions: list[Action] = field(default_factory=list)
    previousLastRobberUserId: int = 0

    def __post_init__(self):
        for p in self.player.game.players:
            if(p.lastRobberUser == True):
                self.previousLastRobberUserId = p.id

    def execute(self):
        self.actions.extend([SetRobberTile(self.tilePosition), StealResourceCommand(self.player, Board.Board().tiles[self.tilePosition])])        
        for action in self.actions:
            action.execute()

        self.player.game.players[self.previousLastRobberUserId-1].lastRobberUser = False
        self.player.lastRobberUser = True

    def undo(self):
        for action in reversed(self.actions):
            action.undo()

        if(self.previousLastRobberUserId != 0):
            self.player.game.players[self.previousLastRobberUserId-1].lastRobberUser = True
            self.player.lastRobberUser = False
        else:
            self.player.lastRobberUser = False

    def redo(self):
        for action in self.actions:
            action.redo()

        self.player.game.players[self.previousLastRobberUserId-1].lastRobberUser = False
        self.player.lastRobberUser = True

    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}'
        for action in self.actions:
            s+=f'\n\t{action}'
        return s

@dataclass
class SevenOnDicesCommand:
    player: Player
    actions: list[Action] = field(default_factory=list)

    def execute(self):
        for pyr in self.player.game.players:
            half = int(pyr.resourceCount()/2)
            if(pyr.resourceCount() >= 7):
                for _ in range(0, half):
                    _, resource = pyr.strategy.chooseParameters(DiscardResourceCommand, self.player)
                    tmp = DiscardResourceCommand(pyr, resource)
                    tmp.execute()
                    self.actions.append(tmp)
        ev, pos = self.player.strategy.chooseParameters(UseRobberCommand, self.player)
        tmp = UseRobberCommand(self.player, pos)
        self.actions.append(tmp)
        tmp.execute()

    def undo(self):
        for action in reversed(self.actions):
            action.undo()
    def redo(self):
        for action in self.actions:
            action.redo()
    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}'
        for action in self.actions:
            s+=f'\n\t{action}'
        return s

@dataclass
class AddResourceToPlayer:
    player: Player
    resource: str

    def execute(self):
        self.player.resources[self.resource] += 1
    def undo(self):
        self.player.resources[self.resource] -= 1
    def redo(self):
        self.player.resources[self.resource] += 1
    def __repr__(self) -> str:
        return f'{self.__class__.__name__} res: {self.resource}'

@dataclass
class RemoveResourceToPlayer:
    player: Player
    resource: str
    unduable: bool = True
    def execute(self):
        # assert self.player.resources[self.resource]>0, f"Player {self.player.id} can't EXECUTE this action becouse it has not {self.resource}"
        if(self.player.resources[self.resource] > 0):
            self.player.resources[self.resource] -= 1
        else:
            self.unduable = False
    def undo(self):
        if(self.unduable):
            self.player.resources[self.resource] += 1
    def redo(self):
        # assert self.player.resources[self.resource]>0, f"Player {self.player.id} can't UNDO this action becouse it has not {self.resource}"
        if(self.player.resources[self.resource] > 0):
            self.player.resources[self.resource] -= 1
        else:
            self.unduable = False
    def __repr__(self) -> str:
        return f'{self.__class__.__name__} res: {self.resource}'

@dataclass
class AddResourceToBank:
    resource: str
    def execute(self):
        Bank.Bank().resources[self.resource] += 1
    def undo(self):
        Bank.Bank().resources[self.resource] -= 1
    def redo(self):
        Bank.Bank().resources[self.resource] += 1
    def __repr__(self) -> str:
        return f'{self.__class__.__name__} res: {self.resource}'

@dataclass
class RemoveResourceToBank:
    resource: str
    def execute(self):
        Bank.Bank().resources[self.resource] -= 1
    def undo(self):
        Bank.Bank().resources[self.resource] += 1
    def redo(self):
        Bank.Bank().resources[self.resource] -= 1
    def __repr__(self) -> str:
        return f'{self.__class__.__name__} res: {self.resource}'


@dataclass
class BankGiveResourceCommand:
    player: Player
    resource: str
    actions: list[Action] = field(default_factory=list)

    def execute(self):
        if self.resource != "desert":
            if Bank.Bank().resources[self.resource] > 0:
                self.actions.append(AddResourceToPlayer(self.player, self.resource))
                self.actions.append(RemoveResourceToBank(self.resource))
            else:
                print(f'Bank does not have {self.resource} anymore. Request by Player {self.player.id}')
        for action in self.actions:
            action.execute()

    def undo(self):
        for action in reversed(self.actions):
            action.undo()
    def redo(self):
        for action in self.actions:
            action.redo()
    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}'
        for action in self.actions:
            s+=f'\n\t{action}'
        return s


@dataclass
class PlayerSpendResourceCommand:
    player: Player
    resource: str
    actions: list[Action] = field(default_factory=list)

    def execute(self):
        self.actions.append(RemoveResourceToPlayer(self.player, self.resource))
        self.actions.append(AddResourceToBank(self.resource))
        for action in self.actions:
            action.execute()
    def undo(self):
        for action in reversed(self.actions):
            action.undo()
    def redo(self):
        for action in self.actions:
            action.redo()
    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}'
        for action in self.actions:
            s+=f'\n\t{action}'
        return s

@dataclass
class DiceProductionCommand:
    game: Game
    dice: int = None
    sevenOnDices: SevenOnDicesCommand = None
    actions: list[Action] = field(default_factory=list)

    def __post_init__(self):
        self.dice = self.game.dices[self.game.actualTurn]

    def execute(self):
        if(self.dice == 7):
            self.actions.append(SevenOnDicesCommand(self.game.currentTurnPlayer))
        else:
            for tile in Board.Board().tiles:
                if tile.number == self.dice and tile != Board.Board().robberTile:
                    for p in tile.associatedPlaces:
                        if(Board.Board().places[p].owner != 0):
                            if(Board.Board().places[p].isColony):
                                self.actions.append(BankGiveResourceCommand(self.game.players[Board.Board().places[p].owner-1], tile.resource))
                            elif(Board.Board().places[p].isCity):
                                self.actions.append(BankGiveResourceCommand(self.game.players[Board.Board().places[p].owner-1], tile.resource))
                                self.actions.append(BankGiveResourceCommand(self.game.players[Board.Board().places[p].owner-1], tile.resource))
        
        for action in self.actions:
            action.execute()
        
    def undo(self):
        for action in reversed(self.actions):
            action.undo()
    def redo(self):
        for action in self.actions:
            action.redo()
    def __repr__(self) -> str:
        s = f'{self.__class__.__name__} dice: {self.dice}'
        for action in self.actions:
            s+=f'\n\t{action}'
        return s

@dataclass
class DefaultPassTurnCommand:
    player: Player
    importantTemp: any = None # serve per prendere il secondo elemento di thing needed (che è un None)
    def execute(self):
        self.player.game.actualTurn += 1
        self.player.game.currentTurnPlayer = self.player.game.players[self.player.game.actualTurn%self.player.game.nplayers]

    def undo(self):
        self.player.game.actualTurn -= 1
        self.player.game.currentTurnPlayer = self.player.game.players[self.player.game.actualTurn%self.player.game.nplayers]

    def redo(self):
        self.player.game.actualTurn += 1
        self.player.game.currentTurnPlayer = self.player.game.players[self.player.game.actualTurn%self.player.game.nplayers]
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'

@dataclass
class PassTurnCommand:
    player: Player
    importantTemp: any = None # serve per prendere il secondo elemento di thing needed (che è un None)
    actions: list[Action] = field(default_factory=list)

    def execute(self):
        self.actions.append(DefaultPassTurnCommand(self.player))
        self.actions.append(InitialTurnSetupCommand(self.player))
        self.actions.append(DiceProductionCommand(self.player.game))
        for action in self.actions:
            action.execute()
        
    def undo(self):
        for action in reversed(self.actions):
            action.undo()
    def redo(self):
        for action in self.actions:
            action.redo()
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'

@dataclass
class PlaceFreeStreetCommand:
    player: Player
    edge: tuple()
    def execute(self):
        Board.Board().edges[self.edge] = self.player.id
        self.player.nStreets+=1
        self.player.ownedStreets.append(self.edge)

    def undo(self):
        Board.Board().edges[self.edge] = 0
        self.player.nStreets-=1
        del self.player.ownedStreets[-1]

    def redo(self):
        self.execute()
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'

@dataclass
class PlaceInitialStreetCommand:
    player: Player
    edge: tuple()
    def execute(self):
        Board.Board().edges[self.edge] = self.player.id
        self.player.nStreets+=1
        self.player.ownedStreets.append(self.edge)

    def undo(self):
        Board.Board().edges[self.edge] = 0
        self.player.nStreets-=1
        del self.player.ownedStreets[-1]

    def redo(self):
        self.execute()
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'

@dataclass
class PlaceInitialColonyCommand:
    player: Player
    place: cg.Place

    def execute(self):
        Board.Board().places[self.place.id].owner = self.player.id
        Board.Board().places[self.place.id].isColony = True
        self.player.victoryPointsModification(1)
        self.player.nColonies+=1
        self.player.ownedColonies.append(self.place.id)
        if(self.place.harbor != ""):
            self.player.ownedHarbors.append(self.place.harbor)
    def undo(self):
        Board.Board().places[self.place.id].owner = 0
        Board.Board().places[self.place.id].isColony = False
        self.player.victoryPointsModification(-1)
        self.player.nColonies-=1
        del self.player.ownedColonies[-1]
        if(self.place.harbor != ""):
            del self.player.ownedHarbors[-1]

    def redo(self):
        self.execute()
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'

@dataclass
class PlaceDefaultCityCommand:
    player: Player
    place: cg.Place

    def execute(self):
        Board.Board().places[self.place.id].isColony = False
        Board.Board().places[self.place.id].isCity = True
        self.player.victoryPointsModification(1)
        self.player.nCities+=1
        self.player.nColonies-=1
        self.player.ownedCities.append(self.place.id)

    def undo(self):
        Board.Board().places[self.place.id].isColony = True
        Board.Board().places[self.place.id].isCity = False
        self.player.victoryPointsModification(-1)
        self.player.nCities-=1
        self.player.nColonies+=1
        del self.player.ownedCities[-1]

    def redo(self):
        self.execute()
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'

@dataclass
class PlaceSecondColonyCommand:
    player: Player
    place: cg.Place
    actions: list[Action] = field(default_factory=list)  

    def execute(self):
        self.actions.append(PlaceInitialColonyCommand(self.player, self.place))
        for touchedResource in Board.Board().places[self.place.id].touchedResourses:
            self.actions.append(BankGiveResourceCommand(self.player, touchedResource))
        
        for action in self.actions:
            action.execute()
        
    def undo(self):
        for action in reversed(self.actions):
            action.undo()
    def redo(self):
        for action in self.actions:
            action.redo()
    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}'
        for action in self.actions:
            s+=f'\n\t{action}'
        return s
    
@dataclass
class FirstChoiseCommand:
    player: Player
    placeChoosen: cg.Place
    actions: list[Action] = field(default_factory=list)

    def execute(self):
        tmp = PlaceInitialColonyCommand(self.player, self.placeChoosen)
        self.actions.append(tmp)
        tmp.execute()
        # print("Res: ", self.player.strategy.chooseParameters(PlaceInitialStreetCommand, self.player))
        _, edgeChoosen = self.player.strategy.chooseParameters(PlaceInitialStreetCommand, self.player)
        # print("475 commands: ",edgeChoosen)
        tmp = PlaceInitialStreetCommand(self.player, edgeChoosen)
        self.actions.append(tmp)
        tmp.execute()
        tmp = DefaultPassTurnCommand(self.player)
        self.actions.append(tmp)
        tmp.execute()        
        
    def undo(self):
        for action in reversed(self.actions):
            action.undo()
    def redo(self):
        for action in self.actions:
            action.redo()
    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}'
        for action in self.actions:
            s+=f'\n\t{action}'
        return s

@dataclass
class SecondChoiseCommand:
    player: Player
    placeChoosen: cg.Place
    actions: list[Action] = field(default_factory=list)

    def execute(self):
        tmp = PlaceSecondColonyCommand(self.player, self.placeChoosen)
        self.actions.append(tmp)
        tmp.execute()
        _, edgeChoosen = self.player.strategy.chooseParameters(PlaceInitialStreetCommand, self.player)
        tmp = PlaceInitialStreetCommand(self.player, edgeChoosen)
        self.actions.append(tmp)
        tmp.execute()
        tmp = DefaultPassTurnCommand(self.player)
        self.actions.append(tmp)
        tmp.execute()
        
    def undo(self):
        for action in reversed(self.actions):
            action.undo()
    def redo(self):
        for action in self.actions:
            action.redo()
    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}'
        for action in self.actions:
            s+=f'\n\t{action}'
        return s

@dataclass
class CheckLongestStreetCommand:
    game: Game
    previousLongestStreetOwner: Player = None
    actualLongestStreetOwner: Player = None
    previousMaxLength: int = None
    actualMaxLength: int = None

    def __post_init__(self):
        self.previousLongestStreetOwner = self.game.longestStreetOwner
        self.previousMaxLength = self.game.longestStreetLength

    def execute(self):
        self.actualLongestStreetOwner, self.actualMaxLength = self.game.longestStreetPlayer()
        if(self.previousLongestStreetOwner != self.actualLongestStreetOwner):
            self.game.longestStreetOwner = self.actualLongestStreetOwner
            self.game.longestStreetLength = self.actualMaxLength
            self.actualLongestStreetOwner.victoryPointsModification(2)
            self.previousLongestStreetOwner.victoryPointsModification(-2)
        self.game.longestStreetLength = self.actualMaxLength

    def undo(self):
        if(self.previousLongestStreetOwner != self.actualLongestStreetOwner):
            self.game.longestStreetOwner = self.previousLongestStreetOwner
            self.game.longestStreetLength = self.previousMaxLength
            self.actualLongestStreetOwner.victoryPointsModification(-2)
            self.previousLongestStreetOwner.victoryPointsModification(2)
        self.game.longestStreetLength = self.previousMaxLength
        
    def redo(self):
        if(self.previousLongestStreetOwner != self.actualLongestStreetOwner):
            self.game.longestStreetOwner = self.actualLongestStreetOwner
            self.game.longestStreetLength = self.previousMaxLength
            self.actualLongestStreetOwner.victoryPointsModification(2)
            self.previousLongestStreetOwner.victoryPointsModification(-2)
        self.game.longestStreetLength = self.actualMaxLength
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'


@dataclass
class PlaceStreetCommand:
    player: Player
    edge: tuple()
    actions: list[Action] = field(default_factory=list)

    def execute(self):

        self.actions.append(PlayerSpendResourceCommand(self.player, "wood"))
        self.actions.append(PlayerSpendResourceCommand(self.player, "clay"))

        self.actions.append(PlaceInitialStreetCommand(self.player, self.edge))
        self.actions.append(CheckLongestStreetCommand(self.player.game))
        for action in self.actions:
            action.execute()
        
    def undo(self):
        for action in reversed(self.actions):
            action.undo()
    def redo(self):
        for action in self.actions:
            action.redo()
    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}'
        for action in self.actions:
            s+=f'\n\t{action}'
        return s

@dataclass
class PlaceColonyCommand:
    player: Player
    place: cg.Place
    actions: list[Action] = field(default_factory=list)

    def execute(self):
        self.actions.extend([PlayerSpendResourceCommand(self.player, "wood"), 
                                PlayerSpendResourceCommand(self.player, "clay"), 
                                PlayerSpendResourceCommand(self.player, "crop"), 
                                PlayerSpendResourceCommand(self.player, "sheep")])
        self.actions.append(PlaceInitialColonyCommand(self.player, self.place))
        self.actions.append(CheckLongestStreetCommand(self.player.game))
        for action in self.actions:
            action.execute()
        
    def undo(self):
        for action in reversed(self.actions):
            action.undo()
    def redo(self):
        for action in self.actions:
            action.redo()
    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}'
        for action in self.actions:
            s+=f'\n\t{action}'
        return s
        

@dataclass
class PlaceCityCommand:
    player: Player
    place: cg.Place
    actions: list[Action] = field(default_factory=list)

    def execute(self):
        self.actions.extend([PlayerSpendResourceCommand(self.player, "iron"),
                                PlayerSpendResourceCommand(self.player, "iron"),
                                PlayerSpendResourceCommand(self.player, "iron"),
                                PlayerSpendResourceCommand(self.player, "crop"),
                                PlayerSpendResourceCommand(self.player, "crop")])

        self.actions.append(PlaceDefaultCityCommand(self.player, self.place))

        for action in self.actions:
            action.execute()
    def undo(self):
        for action in reversed(self.actions):
            action.undo()
    def redo(self):
        for action in self.actions:
            action.redo()
    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}'
        for action in self.actions:
            s+=f'\n\t{action}'
        return s

@dataclass
class BuyDevCardCommand:
    player: Player
    card: str = ""
    actions: list[Action] = field(default_factory=list)

    def execute(self):
        self.actions.extend([PlayerSpendResourceCommand(self.player, "iron"),
                                        PlayerSpendResourceCommand(self.player, "crop"),
                                        PlayerSpendResourceCommand(self.player, "sheep")])
        
        self.player.boughtCards += 1

        for operation in self.actions:
            operation.execute()

        self.card = Board.Board().deck[0]
        if(self.card == "knight"):
            self.player.justBoughtKnights += 1
        if(self.card == "monopoly"):
            self.player.justBoughtMonopolyCard += 1
        if(self.card == "road_building"):
            self.player.justBoughtRoadBuildingCard += 1
        if(self.card == "year_of_plenty"):
            self.player.justBoughtYearOfPlentyCard += 1
        if(self.card == "victory_point"):
            # self.player.victoryPoints += 1
            self.player.victoryPointsModification(1)
            self.player.victoryPointsCards += 1
        Board.Board().deck = Board.Board().deck[1:]

    def undo(self):
        for operation in reversed(self.actions):
            operation.undo()

        self.player.boughtCards -= 1

        if(self.card == "knight"):
            self.player.justBoughtKnights -= 1
        if(self.card == "monopoly"):
            self.player.justBoughtMonopolyCard -= 1
        if(self.card == "road_building"):
            self.player.justBoughtRoadBuildingCard -= 1
        if(self.card == "year_of_plenty"):
            self.player.justBoughtYearOfPlentyCard -= 1
        if(self.card == "victory_point"):
            self.player.victoryPointsModification(-1)
            self.player.victoryPointsCards -= 1
        Board.Board().deck = np.insert(Board.Board().deck, 0, self.card)
        
    def redo(self):
        for operation in self.actions:
            operation.redo()

        self.player.boughtCards += 1

        self.card = Board.Board().deck[0]
        if(self.card == "knight"):
            self.player.justBoughtKnights += 1
        if(self.card == "monopoly"):
            self.player.justBoughtMonopolyCard += 1
        if(self.card == "road_building"):
            self.player.justBoughtRoadBuildingCard += 1
        if(self.card == "year_of_plenty"):
            self.player.justBoughtYearOfPlentyCard += 1
        if(self.card == "victory_point"):
            self.player.victoryPointsModification(1)
            self.player.victoryPointsCards += 1
        Board.Board().deck = Board.Board().deck[1:]
    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}'
        for action in self.actions:
            s+=f'\n\t{action}'
        return s

@dataclass
class DiscardResourceCommand:
    player: Player
    resource: str
    playerSpendResource: PlayerSpendResourceCommand = None

    def execute(self):
        self.playerSpendResource = PlayerSpendResourceCommand(self.player, self.resource)
        self.playerSpendResource.execute()

    def undo(self):
        self.playerSpendResource.undo()

    def redo(self):
        self.playerSpendResource.redo()
    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}'
        s+=f'\n\t{self.playerSpendResource}'
        return s

@dataclass
class StealResourceCommand:
    player: Player
    tile: cg.Tile
    chosenPlayer: Player = None
    takenResource: str = None
    actions: list[Action] = field(default_factory=list)

    def __post_init__(self):
        playersInTile = []
        for place in self.tile.associatedPlaces:
            owner = self.player.game.players[Board.Board().places[place].owner-1]
            if owner not in playersInTile and owner.id != 0 and owner != self.player and owner.resourceCount() > 0: 
                playersInTile.append(owner)
        if len(playersInTile) > 0:
            self.chosenPlayer = playersInTile[random.randint(0,len(playersInTile)-1)]
            self.takenResource = self.chosenPlayer.stealFromMe()
        else:
            self.chosenPlayer = None
            self.takenResource = None

    def execute(self):
        if self.chosenPlayer is not None and self.takenResource is not None:
            self.actions.append(AddResourceToPlayer(self.player, self.takenResource))
            self.actions.append(RemoveResourceToPlayer(self.chosenPlayer, self.takenResource))
        for action in self.actions:
            action.execute()
    def undo(self):
        for action in reversed(self.actions):
            action.undo()
    def redo(self):
        for action in self.actions:
            action.redo()
    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}'
        for action in self.actions:
            s+=f'\n\t{action}'
        return s

@dataclass
class CheckLargestArmyCommand:
    game: Game
    previousLargestArmyOwner: Player = None
    actualLargestArmyOwner: Player = None

    def __post_init__(self):
        self.previousLargestArmyOwner = self.game.largestArmyPlayer

    def execute(self):
        self.actualLargestArmyOwner = self.game.largestArmy()
        if(self.previousLargestArmyOwner != self.actualLargestArmyOwner):
            self.game.largestArmyPlayer = self.actualLargestArmyOwner
            # self.actualLargestArmyOwner.victoryPoints += 2 
            # self.previousLargestArmyOwner.victoryPoints -= 2
            self.actualLargestArmyOwner.victoryPointsModification(2)
            self.previousLargestArmyOwner.victoryPointsModification(-2)
    def undo(self):
        if(self.previousLargestArmyOwner != self.actualLargestArmyOwner):
            self.game.largestArmyPlayer = self.previousLargestArmyOwner
            # self.actualLargestArmyOwner.victoryPoints -= 2 
            # self.previousLargestArmyOwner.victoryPoints += 2
            self.actualLargestArmyOwner.victoryPointsModification(-2)
            self.previousLargestArmyOwner.victoryPointsModification(2)
    def redo(self):
        if(self.previousLargestArmyOwner != self.actualLargestArmyOwner):
            self.game.largestArmyPlayer = self.actualLargestArmyOwner
            # self.actualLargestArmyOwner.victoryPoints += 2 
            # self.previousLargestArmyOwner.victoryPoints -= 2
            self.actualLargestArmyOwner.victoryPointsModification(2)
            self.previousLargestArmyOwner.victoryPointsModification(-2)
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'

@dataclass
class UseKnightCommand:
    player: Player
    tilePosition: cg.Tile
    actions: list[Action] = field(default_factory=list)
    previousLastRobberUserId: int = 0

    def __post_init__(self):
        for p in self.player.game.players:
            if(p.lastRobberUser == True):
                self.previousLastRobberUserId = p.id

    def execute(self):
        self.player.unusedKnights -= 1
        self.player.usedKnights += 1
        self.actions.append(UseRobberCommand(self.player, self.tilePosition))
        self.actions.append(CheckLargestArmyCommand(self.player.game))
        for action in self.actions:
            action.execute()
        self.player.game.players[self.previousLastRobberUserId-1].lastRobberUser = False
        self.player.lastRobberUser = True

    def undo(self):
        for action in reversed(self.actions):
            action.undo()
        self.player.unusedKnights += 1
        self.player.usedKnights -= 1

        if(self.previousLastRobberUserId != 0):
            self.player.game.players[self.previousLastRobberUserId-1].lastRobberUser = True
            self.player.lastRobberUser = False
        else:
            self.player.lastRobberUser = False

    def redo(self):
        self.player.unusedKnights -= 1
        self.player.usedKnights += 1
        for action in self.actions:
            action.redo()   
        self.player.game.players[self.previousLastRobberUserId-1].lastRobberUser = False
        self.player.lastRobberUser = True

    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}'
        for action in self.actions:
            s+=f'\n\t{action}'
        return s 

@dataclass
class TradeBankCommand:
    player: Player
    coupleOfResources: tuple()
    actions: list[Action] = field(default_factory=list)

    def execute(self):
        toTake, toGive = self.coupleOfResources
        self.actions.append(BankGiveResourceCommand(self.player, toTake))
        self.actions.extend([PlayerSpendResourceCommand(self.player, toGive) for _ in range(0, Bank.Bank().resourceToAsk(self.player, toGive))])
        for action in self.actions:
            action.execute()
    def undo(self):
        for action in reversed(self.actions):
            action.undo()
    def redo(self):
        for action in self.actions:
            action.redo()
    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}'
        for action in self.actions:
            s+=f'\n\t{action}'
        return s

@dataclass
class UseMonopolyCardCommand:
    player: Player
    resource: str
    actions: list() = field(default_factory = list)

    def execute(self):
        self.player.monopolyCard -= 1
        for p in self.player.game.players:
            if p != self.player:
                self.actions.extend([RemoveResourceToPlayer(p, self.resource) for _ in range(p.resources[self.resource])])
                self.actions.extend([AddResourceToPlayer(self.player, self.resource) for _ in range(p.resources[self.resource])])
        for action in self.actions:
            action.execute()

    def undo(self):
        self.player.monopolyCard += 1
        for action in self.actions:
            action.undo()
    def redo(self):
        self.player.monopolyCard -= 1
        for action in self.actions:
            action.redo()
    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}'
        for action in self.actions:
            s+=f'\n\t{action}'
        return s

@dataclass
class UseRoadBuildingCardCommand:
    player: Player
    edges: tuple()
    actions: list() = field(default_factory = list)

    def execute(self):
        self.player.roadBuildingCard -= 1
        e1, e2 = self.edges
        if e1 is not None:
            self.actions.append(PlaceInitialStreetCommand(self.player, e1))
        if e2 is not None:
            self.actions.append(PlaceInitialStreetCommand(self.player, e2))
        
        for action in self.actions:
            action.execute()

    def undo(self):
        self.player.roadBuildingCard += 1
        for action in self.actions:
            action.undo()

    def redo(self):
        self.player.roadBuildingCard += 1
        for action in self.actions:
            action.redo()
    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}'
        for action in self.actions:
            s+=f'\n\t{action}'
        return s

@dataclass
class UseYearOfPlentyCardCommand:
    player: Player
    resources: list()
    actions: list[Action] = field(default_factory=list)

    def execute(self):
        self.player.yearOfPlentyCard -= 1
        self.actions.append(BankGiveResourceCommand(self.player, self.resources[0]))
        self.actions.append(BankGiveResourceCommand(self.player, self.resources[1]))
        for action in self.actions:
            action.execute()
    def undo(self):
        self.player.yearOfPlentyCard += 1
        for action in reversed(self.actions):
            action.undo()
    def redo(self):
        self.player.yearOfPlentyCard -= 1
        for action in self.actions:
            action.redo()
    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}'
        for action in self.actions:
            s+=f'\n\t{action}'
        return s

def cardCommands():
    return [UseMonopolyCardCommand, UseKnightCommand, UseYearOfPlentyCardCommand, UseRoadBuildingCardCommand]