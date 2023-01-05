from dataclasses import dataclass, field
import Classes.PlayerWithCommands as Player
import Classes.GameWithCommands as Game
import Classes.Board as Board
import Classes.CatanGraph as cg
import Classes.Bank as Bank
import random
import numpy as np

@dataclass
class PassTurnCommand:
    player: Player
    importantTemp: any = None # serve per prendere il secondo elemento di thing needed (che è un None)
    def execute(self):
        self.player.game.actualTurn += 1
        self.player.game.currentTurnPlayer = self.player.game.players[self.player.game.actualTurn%4]

    def undo(self):
        self.player.game.actualTurn -= 1
        self.player.game.currentTurnPlayer = self.player.game.players[self.player.game.actualTurn%4]

    def redo(self):
        self.player.game.actualTurn += 1
        self.player.game.currentTurnPlayer = self.player.game.players[self.player.game.actualTurn%4]

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

@dataclass
class PlaceInitialColonyCommand:
    player: Player
    place: cg.Place
    isSecond: bool = False

    def execute(self):
        Board.Board().places[self.place.id].owner = self.player.id
        Board.Board().places[self.place.id].isColony = True
        self.player.victoryPoints+=1
        self.player.nColonies+=1
        self.player.ownedColonies.append(self.place.id)
        if(self.place.harbor != ""):
            self.player.ownedHarbors.append(self.place.harbor)
        if(self.isSecond):
            for touchedResource in Board.Board().places[self.place.id].touchedResourses:
                Bank.Bank().giveResource(self.player, touchedResource)

    def undo(self):
        Board.Board().places[self.place.id].owner = 0
        Board.Board().places[self.place.id].isColony = False
        self.player.victoryPoints-=1
        self.player.nColonies-=1
        del self.player.ownedColonies[-1]
        if(self.place.harbor != ""):
            del self.player.ownedHarbors[-1]
        if(self.isSecond):
            for touchedResource in Board.Board().places[self.place.id].touchedResourses:
                self.player.useResource(touchedResource)

    def redo(self):
        self.execute()

@dataclass
class InitialChoiseCommand:
    player: Player
    # place : cg.Place = None
    # edge : tuple() = None
    # isSecond: bool = False
    placeInitialColony: PlaceInitialColonyCommand
    placeInitialStreet: PlaceInitialStreetCommand = None
    passturn: PassTurnCommand = None

    def execute(self):
        #self.placeInitialColony = PlaceInitialColonyCommand(self.player, self.place, self.isSecond)
        #self.placeInitialStreet = PlaceInitialStreetCommand(self.player, self.edge)
        self.placeInitialColony.execute()
        evaluation, edgeChoosen = self.player.evaluate(PlaceInitialStreetCommand)
        self.placeInitialStreet = PlaceInitialStreetCommand(self.player, edgeChoosen)
        self.placeInitialStreet.execute()
        self.passTurnCommand = PassTurnCommand(self.player)
        self.passTurnCommand.execute()

    def undo(self):
        self.placeInitialColony.undo()
        self.placeInitialStreet.undo()
        self.passTurnCommand.undo()

    def redo(self):
        self.execute()


@dataclass
class PlaceStreetCommand:
    player: Player
    edge: tuple()
    withCost: bool = True

    def execute(self):
        if self.withCost:
            self.player.useResource("wood")
            self.player.useResource("clay")
        Board.Board().edges[self.edge] = self.player.id
        self.player.nStreets+=1
        self.player.ownedStreets.append(self.edge)

    def undo(self):
        Bank.Bank().giveResource(self.player, "wood")   
        Bank.Bank().giveResource(self.player, "clay")    
        Board.Board().edges[self.edge] = 0
        self.player.nStreets-=1
        del self.player.ownedStreets[-1]

    def redo(self):
        self.execute()

@dataclass
class PlaceColonyCommand:
    player: Player
    place: cg.Place
    withCost: bool = True

    def execute(self):
        if self.withCost:
            self.player.useResource("wood")
            self.player.useResource("clay")
            self.player.useResource("crop")
            self.player.useResource("sheep")

        self.place.owner = self.player.id
        self.place.isColony = True

        self.player.victoryPoints+=1
        self.player.nColonies+=1
        self.player.ownedColonies.append(self.place.id)

        if(self.place.harbor != ""):
            self.player.ownedHarbors.append(self.place.harbor)

    def undo(self):
        if self.withCost:
            Bank.Bank().giveResource(self.player, "wood")   
            Bank.Bank().giveResource(self.player, "clay")  
            Bank.Bank().giveResource(self.player, "crop")   
            Bank.Bank().giveResource(self.player, "sheep")  

        self.place.owner = 0
        self.place.isColony = False
        self.player.victoryPoints-=1
        self.player.nColonies-=1
        del self.player.ownedColonies[-1]

        if(self.place.harbor != ""):
            del self.player.ownedHarbors[-1]

    def redo(self):
        self.execute()

@dataclass
class PlaceCityCommand:
    player: Player
    place: cg.Place
    withCost: bool = True

    def execute(self):
        if self.withCost:
            self.player.useResource("iron")
            self.player.useResource("iron")
            self.player.useResource("iron")
            self.player.useResource("crop")
            self.player.useResource("crop")

        Board.Board().places[self.place.id].isColony = False
        Board.Board().places[self.place.id].isCity = True
        self.player.victoryPoints+=1
        self.player.nCities+=1
        self.player.nColonies-=1
        self.player.ownedCities.append(self.place.id)

    def undo(self):
        if self.withCost:
            Bank.Bank().giveResource(self.player, "iron")
            Bank.Bank().giveResource(self.player, "iron")
            Bank.Bank().giveResource(self.player, "iron")
            Bank.Bank().giveResource(self.player, "crop")
            Bank.Bank().giveResource(self.player, "crop")

        Board.Board().places[self.place.id].isColony = True
        Board.Board().places[self.place.id].isCity = False
        self.player.victoryPoints-=1
        self.player.nCities-=1
        self.player.nColonies+=1
        del self.player.ownedCities[-1]

@dataclass
class BuyDevCardCommand:
    player: Player
    card: str = ""

    def execute(self):
        self.player.useResource("iron")
        self.player.useResource("crop")
        self.player.useResource("sheep")

        self.card = Board.Board().deck[0] ##### IL DECK VIENE TOCCATO QUA

        if(self.card == "knight"):
            self.player.justBoughtKnights += 1
        if(self.card == "monopoly"):
            self.player.justBoughtMonopolyCard += 1
        if(self.card == "road_building"):
            self.player.justBoughtRoadBuildingCard += 1
        if(self.card == "year_of_plenty"):
            self.player.justBoughtYearOfPlentyCard += 1
        if(self.card == "victory_point"):
            self.player.victoryPoints += 1
            self.player.victoryPointsCards += 1
        Board.Board().deck = Board.Board().deck[1:]

    def undo(self):
        Bank.Bank().giveResource(self.player, "iron")
        Bank.Bank().giveResource(self.player, "crop")
        Bank.Bank().giveResource(self.player, "sheep")
        if(self.card == "knight"):
            self.player.justBoughtKnights -= 1
        if(self.card == "monopoly"):
            self.player.justBoughtMonopolyCard -= 1
        if(self.card == "road_building"):
            self.player.justBoughtRoadBuildingCard -= 1
        if(self.card == "year_of_plenty"):
            self.player.justBoughtYearOfPlentyCard -= 1
        if(self.card == "victory_point"):
            self.player.victoryPoints -= 1
            self.player.victoryPointsCards -= 1
        Board.Board().deck = np.insert(Board.Board().deck, 0, self.card)
        
    def redo(self):
        self.execute()

@dataclass
class DiscardResourceCommand:
    player: Player
    resource: str

    def execute(self):
        self.player.useResource(self.resource)

    def undo(self):
        Bank.Bank().giveResource(self.player, self.resource)

    def redo(self):
        self.execute()

def stealResource(player, tile: cg.Tile):
    playersInTile = []
    chosenPlayer = None
    takenResource = None
    for place in tile.associatedPlaces:
        owner = player.game.players[Board.Board().places[place].owner-1]
        if owner not in playersInTile and owner.id != 0 and owner != player and owner.resourceCount() > 0: 
            playersInTile.append(owner)
    if len(playersInTile) > 0:
        chosenPlayer = playersInTile[random.randint(0,len(playersInTile)-1)]
        takenResource = chosenPlayer.stealFromMe(player)
    return chosenPlayer, takenResource

@dataclass
class StealResourceCommand:
    player: Player
    tile: cg.Tile
    chosenPlayer: Player = None # Player.Player(0, Game.Game())
    takenResource: str = None

    def __post_init__(self):
        self.chosenPlayer = self.player.game.dummy

    def execute(self):
        playersInTile = []
        for place in self.tile.associatedPlaces:
            owner = self.player.game.players[Board.Board().places[place].owner-1]
            if owner not in playersInTile and owner.id != 0 and owner != self.player and owner.resourceCount() > 0: 
                playersInTile.append(owner)
        if len(playersInTile) > 0:
            self.chosenPlayer = playersInTile[random.randint(0,len(playersInTile)-1)]
            self.takenResource = self.chosenPlayer.stealFromMe(self.player)
        return

    def undo(self):
        if self.chosenPlayer is not None and self.takenResource is not None:
            self.chosenPlayer.resources[self.takenResource] += 1
            self.player.resources[self.takenResource] -= 1

    def redo(self):
        self.chosenPlayer.resources[self.takenResource] -= 1
        self.player.resources[self.takenResource] += 1

@dataclass
class UseRobberCommand:
    player: Player
    tilePosition: int
    previousPosition: int = 0
    chosenPlayer: Player = None # Player.Player(0, Game.Game())
    takenResource: str = ""
    srCommand = None

    def __post_init__(self):
        self.chosenPlayer = self.player.game.dummy

    def execute(self):
        self.previousPosition = Board.Board().robberTile        
        Board.Board().robberTile = self.tilePosition
        #self.chosenPlayer, self.takenResource = stealResource(self.player, Board.Board().tiles[self.tilePosition])
        # self.srCommand = StealResourceCommand(self.player, Board.Board().tiles[self.tilePosition])
        # self.srCommand.execute()

    def undo(self):
        Board.Board().robberTile = self.previousPosition
        # if self.chosenPlayer is not None and self.takenResource is not None:
        #     self.chosenPlayer.resources[self.takenResource] += 1
        #     self.player.resources[self.takenResource] -= 1
        #self.srCommand.undo()
    
    def redo(self):
        self.execute()

@dataclass
class UseKnightCommand:
    player: Player
    tilePosition: cg.Tile
    previousPosition: int = 0
    previousLargestArmy : Player = None # Player.Player(0, Game.Game())
    postMoveLargArmy : Player = None #Player.Player(0, Game.Game())
    chosenPlayer: Player = None #Player.Player(0, Game.Game())
    takenResource: str = ""

    def __post_init__(self):
        self.previousLargestArmy = self.player.game.dummy
        self.postMoveLargArmy = self.player.game.dummy
        self.chosenPlayer = self.player.game.dummy

    def execute(self):
        self.previousLargestArmy = self.player.game.largestArmy()   
        self.previousPosition = Board.Board().robberTile
        Board.Board().robberTile = self.tilePosition
        #self.stealCommand = StealResourceCommand(self.player, Board.Board().tiles[self.tilePosition])
        #self.chosenPlayer, self.takenResource = stealResource(self.player, Board.Board().tiles[self.tilePosition])
        self.player.unusedKnights -= 1
        self.player.usedKnights += 1

        self.postMoveLargArmy = self.player.game.largestArmy()

        self.postMoveLargArmy.victoryPoints += 2 
        self.previousLargestArmy.victoryPoints -= 2

    def undo(self):
        self.player.game.largestArmyPlayer = self.previousLargestArmy

        self.player.unusedKnights += 1
        self.player.usedKnights -= 1

        Board.Board().robberTile = self.previousPosition

        self.postMoveLargArmy.victoryPoints -= 2 
        self.previousLargestArmy.victoryPoints += 2

        # if self.chosenPlayer is not None and self.takenResource is not None:
        #     self.chosenPlayer.resources[self.takenResource] += 1
        #     self.player.resources[self.takenResource] -= 1

    def redo(self):
        self.execute()

@dataclass
class TradeBankCommand:
    player: Player
    coupleOfResources: tuple()

    def execute(self):
        toTake, toGive = self.coupleOfResources
        Bank.Bank().giveResource(self.player, toTake)
        #print(self.player.id, " take ", toTake, " from the bank. ")

        for _ in range(0, Bank.Bank().resourceToAsk(self.player, toGive)):
            self.player.useResource(toGive)
            #print(self.player.id, " give ", toGive, "to the bank. ", _)

    def undo(self):
        toTake, toGive = self.coupleOfResources
        self.player.useResource(toTake)
        #print(self.player.id, " give ", toTake, "to the bank. ")

        for _ in range(0, Bank.Bank().resourceToAsk(self.player, toGive)):
            Bank.Bank().giveResource(self.player, toGive)
            #print(self.player.id, " take ", toGive, " from the bank. ", _)

    def redo(self):
        self.execute()

@dataclass
class UseMonopolyCardCommand:
    player: Player
    resource: str
    previousPlayersResources: list() = field(default_factory = list) # [2,3,4,5] -> [0,14,0,0]

    def execute(self):
        self.player.monopolyCard -= 1
        sum = 0
        for p in self.player.game.players:
            sum = sum + p.resources[self.resource]
            self.previousPlayersResources.append(p.resources[self.resource])
            p.resources[self.resource] = 0
        self.player.resources[self.resource] = sum
        return

    def undo(self):
        self.player.monopolyCard += 1
        for i, p in enumerate(self.player.game.players):
            p.resources[self.resource] = self.previousPlayersResources[i]

    def redo(self):
        self.execute()

@dataclass
class UseRoadBuildingCardCommand:
    player: Player
    edges: tuple()
    placeStreetCommand1: PlaceStreetCommand = None
    placeStreetCommand2: PlaceStreetCommand = None

    def execute(self):
        self.player.roadBuildingCard -= 1
        e1, e2 = self.edges
        if e1 is not None:
            self.placeStreetCommand1 = PlaceStreetCommand(self.player, e1, False)
            self.placeStreetCommand1.execute()

        if e2 is not None:
            self.placeStreetCommand2 = PlaceStreetCommand(self.player, e2, False)
            self.placeStreetCommand2.execute()

    def undo(self):
        self.player.roadBuildingCard += 1
        self.placeStreetCommand1.undo()
        self.placeStreetCommand2.undo()

    def redo(self):
        self.execute()

@dataclass
class UseYearOfPlentyCardCommand:
    player: Player
    resources: list()

    def execute(self):
        self.player.yearOfPlentyCard -= 1
        Bank.Bank().giveResource(self.player, self.resources[0])
        Bank.Bank().giveResource(self.player, self.resources[1])

    def undo(self):
        self.player.yearOfPlentyCard += 1
        self.player.useResource(self.resources[0])
        self.player.useResource(self.resources[1])

    def redo(self):
        self.execute()

@dataclass
class DiceProductionCommand:
    number: int
    game: Game

    def execute(self):
        for tile in Board.Board().tiles:
            if tile.number == self.number and tile != Board.Board().robberTile:
                for p in tile.associatedPlaces:
                    if(Board.Board().places[p].owner != 0):
                        if(Board.Board().places[p].isColony):
                            Bank.Bank().giveResource(self.game.players[Board.Board().places[p].owner-1], tile.resource)
                        elif(Board.Board().places[p].isCity):
                            Bank.Bank().giveResource(self.game.players[Board.Board().places[p].owner-1], tile.resource)
                            Bank.Bank().giveResource(self.game.players[Board.Board().places[p].owner-1], tile.resource)

    def undo(self):
        for tile in Board.Board().tiles:
            if tile.number == self.number and tile != Board.Board().robberTile:
                for p in tile.associatedPlaces:
                    if(Board.Board().places[p].owner != 0):
                        if(Board.Board().places[p].isColony):
                            self.game.players[Board.Board().places[p].owner-1].useResource(tile.resource)
                        elif(Board.Board().places[p].isCity):
                            self.game.players[Board.Board().places[p].owner-1].useResource(tile.resource)
                            self.game.players[Board.Board().places[p].owner-1].useResource(tile.resource)

    def redo(self):
        self.execute()

@dataclass
class InitialTurnSetupCommand:
    player: Player #c.PlayerWithCommands

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

        #global devCardsBought
        #turnCardUsed = False 
        self.player.unusedKnights = self.player.unusedKnights + self.player.justBoughtKnights
        self.player.justBoughtKnights = 0
        self.player.monopolyCard += self.player.justBoughtMonopolyCard
        self.player.justBoughtMonopolyCard = 0
        self.player.roadBuildingCard += self.player.justBoughtRoadBuildingCard
        self.player.justBoughtRoadBuildingCard = 0
        self.player.yearOfPlentyCard += self.player.justBoughtYearOfPlentyCard
        self.player.justBoughtYearOfPlentyCard = 0
        self.player.turnCardUsed = False
        #view.updateGameScreen() 
        #dicesValue = self.player.game.dices[self.player.game.actualTurn]
        # if(dicesValue == 7):
        #     self.player.game.sevenOnDices()
        #     ev, pos = self.player.evaluate(UseRobberCommand)
        #     decisionManager(player, commands.UseRobberCommand(player, pos))
        #     goNext()
        #     decisionManager(player, commands.StealResourceCommand(player, c.Board.Board().tiles[pos]))

        #     saveMove(save, player) 
        # else:
        #     decisionManager(player, commands.DiceProductionCommand(dicesValue, game))
        #     view.updateGameScreen()
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

def cardCommands():
    return [UseMonopolyCardCommand, UseKnightCommand, UseYearOfPlentyCardCommand, UseRoadBuildingCardCommand]



