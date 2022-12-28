import Classes.Player as Player
import Classes.Board as Board
import Classes.CatanGraph as cg

@dataclass
class PlaceStreetCommand:
    player: Player
    edge: (int, int)
    withCost: bool

    def execute(self):
        if self.withCost:
            self.player.useResource("wood")
            self.player.useResource("clay")
        Board.Board().edges[edge] = self.player.id
        self.player.nStreets+=1
        self.player.ownedStreets.append(edge)

    def undo(self):
        Bank.Bank().giveResource(self.player, "wood")   
        Bank.Bank().giveResource(self.player, "clay")    
        Board.Board().edges[edge] = 0
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
class placeColonyCommand:
    withCost: bool
    player: Player
    place: Place

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

        Board.Board().places[place.id].isColony = False
        Board.Board().places[place.id].isCity = True
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

        Board.Board().places[place.id].isColony = True
        Board.Board().places[place.id].isCity = False
        self.player.victoryPoints-=1
        self.player.nCities-=1
        self.player.nColonies+=1
        del self.player.ownedCities[-1]

@dataclass
class BuyDevCardCommand:
    withCost: bool
    player: Player

    def execute(self):
        if self.withCost:
            self.player.useResource("iron")
            self.player.useResource("crop")
            self.player.useResource("sheep")

        card = Board.Board().deck[0] ##### IL DECK VIENE TOCCATO QUA

        if(card == "knight"):
            self.player.justBoughtKnights += 1
        if(card == "monopoly"):
            self.player.justBoughtMonopolyCard += 1
        if(card == "road_building"):
            self.player.justBoughtRoadBuildingCard += 1
        if(card == "year_of_plenty"):
            self.player.justBoughtYearOfPlentyCard += 1
        if(card == "victory_point"):
            self.player.victoryPoints += 1
            self.player.victoryPointsCards += 1
        Board.Board().deck = Board.Board().deck[1:] 

    def undo(self):
        print("Debug: BUG wrong way. (riga 114 move")
        if self.withCost:
            Bank.Bank().giveResource(self.player, "iron")
            Bank.Bank().giveResource(self.player, "crop")
            Bank.Bank().giveResource(self.player, "sheep")

    def redo(self):
        self.execute()

@dataclass
class DiscardResourceCommand:
    player: Player
    resource: String
    withCost: bool

    def execute(self):
        if self.withCost:
            self.player.useResource(self.resource)

    def undo(self):
        if self.withCost:
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
    chosenPlayer: Player
    takenResource: String

    def execute(self):
        playersInTile = []
        for place in tile.associatedPlaces:
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
    previousPosition: int
    stealCommand: StealResourceCommand

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
    previousPosition: int
    stealCommand: StealResourceCommand

    def execute(self):
        #largestArmy = player.game.largestArmy(justCheck)   

        self.previousPosition = Board.Board().robberTile
        Board.Board().robberTile = self.tilePosition
        self.stealCommand = StealResourceCommand(self.player, Board.Board().tiles[self.tilePosition])
        self.stealCommand.execute()  
        self.player.unusedKnights -= 1
        self.player.usedKnights += 1

        #postMoveLargArmy = player.game.largestArmy(justCheck)

        # if(largestArmy != postMoveLargArmy):
        #     postMoveLargArmy.victoryPoints += 2 
        #     #print("-2 riga 207")
        #     largestArmy.victoryPoints -= 2

    def undo(self):
        self.player.unusedKnights += 1
        self.player.usedKnights -= 1
        Board.Board().robberTile = self.previousPosition
        self.stealCommand.undo()

    def redo(self):
        self.previousPosition = Board.Board().robberTile
        Board.Board().robberTile = self.tilePosition
        self.stealCommand.redo() 
        self.player.unusedKnights -= 1
        self.player.usedKnights += 1

@dataclass
class TradeBankCommand:
    player: Player
    coupleOfResources: (String, String)

    def execute(self):
        toTake, toGive = coupleOfResources
        Bank.Bank().giveResource(self.player, toTake)
        for _ in range(0, Bank.Bank().resourceToAsk(self.player, toGive)):
            self.player.useResource(toGive)

    def undo(self):
        self.player.useResource(toTake)
        for i in range(0, Bank.Bank().resourceToAsk(self.player, toGive)):
            Bank.Bank().giveResource(self.player, toGive)

    def redo(self):
        self.execute()

@dataclass
class UseMonopolyCardCommand:
    player: Player
    resource: String
    previousPlayersResources: [int]

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




