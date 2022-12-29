from dataclasses import dataclass, field
import Classes.PlayerWithCommands as Player
import Classes.GameWithCommands as Game
import Classes.Board as Board
import Classes.CatanGraph as cg
import Classes.Bank as Bank
import random

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

    #previousLongestStreetOwner = player.game.longestStreetPlayer(justCheck)
    # actualLongestStreetOwner = player.game.longestStreetPlayer(justCheck)
    # if(previousLongestStreetOwner != actualLongestStreetOwner):
    #     player.game.longestStreetOwner = actualLongestStreetOwner
    #     actualLongestStreetOwner.victoryPoints += 2
    #     #print("-2 riga 21")
    #     previousLongestStreetOwner.victoryPoints -= 2

@dataclass
class PlaceInitialColonyCommand:
    player: Player
    place: cg.Place

    def execute(self):
        Board.Board().places[self.place.id].owner = self.player.id
        Board.Board().places[self.place.id].isColony = True
        self.player.victoryPoints+=1
        self.player.nColonies+=1
        self.player.ownedColonies.append(self.place.id)
        if(self.place.harbor != ""):
            self.player.ownedHarbors.append(self.place.harbor)

    def undo(self):
        Board.Board().places[self.place.id].owner = 0
        Board.Board().places[self.place.id].isColony = False
        self.player.victoryPoints-=1
        self.player.nColonies-=1
        del self.player.ownedColonies[-1]
        if(self.place.harbor != ""):
            del self.player.ownedHarbors[-1]

    def redo(self):
        self.execute()

@dataclass
class PlaceStreetCommand:
    player: Player
    edge: tuple()
    withCost: bool

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

        #previousLongestStreetOwner = player.game.longestStreetPlayer(justCheck)
        #actualLongestStreetOwner = player.game.longestStreetPlayer(justCheck)

        # if(previousLongestStreetOwner.id != actualLongestStreetOwner.id): 
        #     player.game.longestStreetOwner = actualLongestStreetOwner
        #     actualLongestStreetOwner.victoryPoints += 2
            #previousLongestStreetOwner.victoryPoints -= 2

@dataclass
class PlaceColonyCommand:
    withCost: bool
    player: Player
    place: cg.Place

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
    withCost: bool
    player: Player
    place: cg.Place

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
    withCost: bool
    player: Player
    card: str

    def execute(self):
        if self.withCost:
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
        if self.withCost:
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
        Board.Board().deck.insert(0, self.card)
        
    def redo(self):
        self.execute()

@dataclass
class DiscardResourceCommand:
    player: Player
    resource: str
    withCost: bool

    def execute(self):
        self.player.useResource(self.resource)

    def undo(self):
        Bank.Bank().giveResource(self.player, self.resource)

    def redo(self):
        self.execute()

@dataclass
class PassTurnCommand:
    def execute(self):
        pass

    def undo(self):
        pass

    def redo(self):
        pass

@dataclass
class StealResourceCommand:
    player: Player
    tile: cg.Tile
    chosenPlayer: Player = field(default_factory = Player.Player(0, Game.Game()))
    takenResource: str = field(default_factory = "")

    def execute(self):
        playersInTile = []
        for place in self.tile.associatedPlaces:
            owner = self.player.game.players[Board.Board().places[place].owner-1]
            if owner not in playersInTile and owner.id != 0 and owner != self.player and owner.resourceCount() > 0: 
                playersInTile.append(owner)
        if len(playersInTile) > 0:
            chosenPlayer = playersInTile[random.randint(0,len(playersInTile)-1)]
            self.takenResource = chosenPlayer.stealFromMe(self.player)
        return

    def undo(self):
        self.chosenPlayer.resources[self.takenResource] += 1
        self.player.resources[self.takenResource] -= 1

    def redo(self):
        self.chosenPlayer.resources[self.takenResource] -= 1
        self.player.resources[self.takenResource] += 1

@dataclass
class UseRobberCommand:
    player: Player
    tilePosition: int
    previousPosition: int = field(default_factory = 0)
    stealCommand: StealResourceCommand = None

    def execute(self):
        self.previousPosition = Board.Board().robberTile
        self.stealCommand = StealResourceCommand(self.player, Board.Board().tiles[self.tilePosition])
        self.stealCommand.execute()        
        Board.Board().robberTile = self.tilePosition

    def undo(self):
        Board.Board().robberTile = self.previousPosition
        self.stealCommand.undo()
    
    def redo(self):
        self.execute()

@dataclass
class UseKnightCommand:
    player: Player
    tilePosition: cg.Tile
    previousPosition: int = field(default_factory = 0)
    stealCommand: StealResourceCommand = None
    previousLargestArmy : Player = field(default_factory = Player.Player(0, Game.Game()))
    postMoveLargArmy : Player = field(default_factory = Player.Player(0, Game.Game())) 

    def execute(self):

        self.previousLargestArmy = self.player.game.largestArmy()   

        self.previousPosition = Board.Board().robberTile
        Board.Board().robberTile = self.tilePosition
        self.stealCommand = StealResourceCommand(self.player, Board.Board().tiles[self.tilePosition])
        self.stealCommand.execute()  
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

        self.stealCommand.undo()

    def redo(self):
        self.execute()

@dataclass
class TradeBankCommand:
    player: Player
    coupleOfResources: tuple()

    def execute(self):
        toTake, toGive = self.coupleOfResources
        Bank.Bank().giveResource(self.player, toTake)
        for _ in range(0, Bank.Bank().resourceToAsk(self.player, toGive)):
            self.player.useResource(toGive)

    def undo(self):
        toTake, toGive = self.coupleOfResources
        self.player.useResource(toTake)
        for i in range(0, Bank.Bank().resourceToAsk(self.player, toGive)):
            Bank.Bank().giveResource(self.player, toGive)

    def redo(self):
        self.execute()

@dataclass
class UseMonopolyCardCommand:
    player: Player
    resource: str
    previousPlayersResources: list() # [2,3,4,5] -> [0,14,0,0]

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
        for p, i in enumerate(self.player.game.players):
            p.resources[self.resource] = self.previousPlayersResources[i]

    def redo(self):
        self.execute()

@dataclass
class UseRoadBuildingCardCommand:
    player: Player
    edges: tuple()
    placeStreetCommand: PlaceStreetCommand

    def execute(self):
        self.player.roadBuildingCard -= 1
        e1, e2 = self.edges
        if e1 is not None:
            self.placeStreetCommand = PlaceStreetCommand(self.player, e1, False)
            self.placeStreetCommand.execute()

        if e2 is not None:
            self.placeStreetCommand = PlaceStreetCommand(self.player, e2, False)
            self.placeStreetCommand.execute()

    def undo(self):
        self.player.roadBuildingCard += 1
        self.placeStreetCommand.undo()

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


def cardCommands():
    return [UseMonopolyCardCommand, UseKnightCommand, UseYearOfPlentyCardCommand, UseRoadBuildingCardCommand]



