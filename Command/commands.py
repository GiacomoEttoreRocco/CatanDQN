from Classes.Player import Player
import CatanGraph as cg

@dataclass
class PlaceStreetCommand:
    player: Player
    edge: (int, int)
    withCost: bool

    def execute(self):
        if withCost:
            player.useResource("wood")
            player.useResource("clay")
        Board.Board().edges[edge] = player.id
        player.nStreets+=1
        player.ownedStreets.append(edge)

    def undo(self):
        Bank.Bank().giveResource(player, "wood")   
        Bank.Bank().giveResource(player, "clay")    
        Board.Board().edges[edge] = 0
        player.nStreets-=1
        del player.ownedStreets[-1]

    def redo(self):
        self.execute()

        #previousLongestStreetOwner = player.game.longestStreetPlayer(justCheck)
        #actualLongestStreetOwner = player.game.longestStreetPlayer(justCheck)

        # if(previousLongestStreetOwner.id != actualLongestStreetOwner.id): 
        #     player.game.longestStreetOwner = actualLongestStreetOwner
        #     actualLongestStreetOwner.victoryPoints += 2
            #previousLongestStreetOwner.victoryPoints -= 2

@dataclass
class placeColonyCommand:
    withCost: bool
    player: Player
    place: Place
    def execute(self):
        if withCost:
            player.useResource("wood")
            player.useResource("clay")
            player.useResource("crop")
            player.useResource("sheep")

        place.owner = player.id
        place.isColony = True

        player.victoryPoints+=1
        player.nColonies+=1
        player.ownedColonies.append(place.id)

        if(place.harbor != ""):
            player.ownedHarbors.append(place.harbor)

    def undo(self):
        if withCost:
            Bank.Bank().giveResource(player, "wood")   
            Bank.Bank().giveResource(player, "clay")  
            Bank.Bank().giveResource(player, "crop")   
            Bank.Bank().giveResource(player, "sheep")  

        place.owner = 0
        place.isColony = False
        player.victoryPoints-=1
        player.nColonies-=1
        del player.ownedColonies[-1]

        if(place.harbor != ""):
            del player.ownedHarbors[-1]

    def redo(self):
        self.execute()

@dataclass
class PlaceCityCommand:
    withCost: bool
    player: Player
    place: cg.Place

    def execute(self):
        if withCost:
            player.useResource("iron")
            player.useResource("iron")
            player.useResource("iron")
            player.useResource("crop")
            player.useResource("crop")

        Board.Board().places[place.id].isColony = False
        Board.Board().places[place.id].isCity = True
        player.victoryPoints+=1
        player.nCities+=1
        player.nColonies-=1
        player.ownedCities.append(place.id)

    def undo(self):
        if withCost:
            Bank.Bank().giveResource(player, "iron")
            Bank.Bank().giveResource(player, "iron")
            Bank.Bank().giveResource(player, "iron")
            Bank.Bank().giveResource(player, "crop")
            Bank.Bank().giveResource(player, "crop")

        Board.Board().places[place.id].isColony = True
        Board.Board().places[place.id].isCity = False
        player.victoryPoints-=1
        player.nCities-=1
        player.nColonies+=1
        del player.ownedCities[-1]

@dataclass
class BuyDevCardCommand:
    withCost: bool
    player: Player

    def execute(self):
        if withCost:
            player.useResource("iron")
            player.useResource("crop")
            player.useResource("sheep")

        card = Board.Board().deck[0] ##### IL DECK VIENE TOCCATO QUA

        if(card == "knight"):
            player.justBoughtKnights += 1
        if(card == "monopoly"):
            player.justBoughtMonopolyCard += 1
        if(card == "road_building"):
            player.justBoughtRoadBuildingCard += 1
        if(card == "year_of_plenty"):
            player.justBoughtYearOfPlentyCard += 1
        if(card == "victory_point"):
            player.victoryPoints += 1
            player.victoryPointsCards += 1
        Board.Board().deck = Board.Board().deck[1:] 

    def undo(self):
        print("Debug: BUG wrong way. (riga 114 move")
        if withCost:
            Bank.Bank().giveResource(player, "iron")
            Bank.Bank().giveResource(player, "crop")
            Bank.Bank().giveResource(player, "sheep")

    def redo(self):
        self.execute()

@dataclass
class DiscardResourceCommand:
    player: Player
    resource: String
    withCost: bool

    def execute(self):
        if withCost:
            player.useResource(resource)

    def undo(self):
        if withCost:
            Bank.Bank().giveResource(player, resource)

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
    chosenPlayer: Player
    takenResource: String

    def execute(self):
        playersInTile = []
        for place in tile.associatedPlaces:
            owner = player.game.players[Board.Board().places[place].owner-1]
            if owner not in playersInTile and owner.id != 0 and owner != player and owner.resourceCount() > 0: 
                playersInTile.append(owner)
        if len(playersInTile) > 0:
            chosenPlayer = playersInTile[random.randint(0,len(playersInTile)-1)]
            self.takenResource = chosenPlayer.stealFromMe(player)
        return

    def undo(self):
        chosenPlayer.resources[self.takenResource] += 1
        player.resources[self.takenResource] -= 1

    def redo(self):
        chosenPlayer.resources[self.takenResource] -= 1
        player.resources[self.takenResource] += 1

@dataclass
class UseRobberCommand:
    player: Player
    tilePosition: int
    previousPosition: int
    stealCommand: StealResourceCommand

    def execute(self):
        self.previousPosition = Board.Board().robberTile
        self.stealCommand = StealResourceCommand(player, Board.Board().tiles[tilePosition])
        self.stealCommand.execute()        
        Board.Board().robberTile = tilePosition

    def undo(self):
        Board.Board().robberTile = self.previousPosition
        self.stealCommand.undo()
    
    def redo(self):
        self.execute()

@dataclass
class UseKnightCommand:
    player: Player
    tilePosition: cg.Tile
    previousPosition: int
    stealCommand: StealResourceCommand

    def execute(self):
        #largestArmy = player.game.largestArmy(justCheck)   

        self.previousPosition = Board.Board().robberTile
        Board.Board().robberTile = tilePosition
        self.stealCommand = StealResourceCommand(player, Board.Board().tiles[tilePosition])
        self.stealCommand.execute()  
        player.unusedKnights -= 1
        player.usedKnights += 1

        #postMoveLargArmy = player.game.largestArmy(justCheck)

        # if(largestArmy != postMoveLargArmy):
        #     postMoveLargArmy.victoryPoints += 2 
        #     #print("-2 riga 207")
        #     largestArmy.victoryPoints -= 2

    def undo(self):
        player.unusedKnights += 1
        player.usedKnights -= 1
        Board.Board().robberTile = self.previousPosition
        self.stealCommand.undo()

    def redo(self):
        self.previousPosition = Board.Board().robberTile
        Board.Board().robberTile = tilePosition
        self.stealCommand.redo() 
        player.unusedKnights -= 1
        player.usedKnights += 1

@dataclass
class TradeBankCommand:
    player: Player
    coupleOfResources: (String, String)

    def execute(self):
        toTake, toGive = coupleOfResources
        Bank.Bank().giveResource(player, toTake)
        for _ in range(0, Bank.Bank().resourceToAsk(player, toGive)):
            player.useResource(toGive)

    def undo(self):
        player.useResource(toTake)
        for i in range(0, Bank.Bank().resourceToAsk(player, toGive)):
            Bank.Bank().giveResource(player, toGive)

    def redo(self):
        self.execute()

@dataclass
class UseMonopolyCardCommand:
    player: Player
    resource: String
    previousPlayersResources: [int]

    def execute(self):
        player.monopolyCard -= 1
        sum = 0
        for p in player.game.players:
            sum = sum + p.resources[resource]
            self.previousPlayersResources.append(p.resources[self.resource])
            p.resources[resource] = 0
        player.resources[resource] = sum
        return

    def undo(self):
        player.monopolyCard += 1
        for p, i in enumerate(player.game.players):
            p.resources[self.resource] = self.previousPlayersResources[i]


    def redo(self):
        self.execute()

@dataclass
class useRoadBuildingCardCommand:
    player: Player
    edges: ((int, int),(int, int))
    placeStreetCommand: PlaceStreetCommand

    def execute(self):
        self.player.roadBuildingCard -= 1
        e1, e2 = edges
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
    resources: [String]

    def execute(self):
        self.player.yearOfPlentyCard -= 1
        Bank.Bank().giveResource(self.player, resources[0])
        Bank.Bank().giveResource(self.player, resources[1])

    def undo(self):
        self.player.yearOfPlentyCard += 1
        self.player.useResource(resources[0])
        self.player.useResource(resources[1])

    def redo(self):
        self.execute()




