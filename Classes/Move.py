import Classes.Board as Board
import Classes.Bank as Bank
import Classes.Player as Player
import Classes.CatanGraph as cg
import random

def placeInitialStreet(player, edge, undo = False, justCheck = False):
    previousLongestStreetOwner = player.game.longestStreetPlayer(justCheck)
    if(not undo):
        Board.Board().edges[edge] = player.id
        player.nStreets+=1
        player.ownedStreets.append(edge)
    else:
        Board.Board().edges[edge] = 0
        player.nStreets-=1
        del player.ownedStreets[-1]

    actualLongestStreetOwner = player.game.longestStreetPlayer(justCheck)
    if(previousLongestStreetOwner != actualLongestStreetOwner):
        player.game.longestStreetOwner = actualLongestStreetOwner
        actualLongestStreetOwner.victoryPoints += 2
        #print("-2 riga 21")
        previousLongestStreetOwner.victoryPoints -= 2

def placeFreeStreet(player, edge, undo = False, justCheck = False):
    previousLongestStreetOwner = player.game.longestStreetOwner
    if(not undo):
        Board.Board().edges[edge] = player.id
        player.nStreets+=1
        player.ownedStreets.append(edge)
    else:
        Board.Board().edges[edge] = 0
        player.nStreets-=1
        del player.ownedStreets[-1]

    actualLongestStreetOwner = player.game.longestStreetPlayer(justCheck)
    if(previousLongestStreetOwner != actualLongestStreetOwner):
        player.game.longestStreetOwner = actualLongestStreetOwner
        actualLongestStreetOwner.victoryPoints += 2
        #print("-2 riga 38, JustCheck(?) ", justCheck)
        previousLongestStreetOwner.victoryPoints -= 2

def placeInitialColony(player: Player, place: cg.Place, undo = False):
    if(not undo):
        Board.Board().places[place.id].owner = player.id
        Board.Board().places[place.id].isColony = True
        player.victoryPoints+=1
        player.nColonies+=1
        player.ownedColonies.append(place.id)
        if(place.harbor != ""):
            player.ownedHarbors.append(place.harbor)
    else:
        Board.Board().places[place.id].owner = 0
        Board.Board().places[place.id].isColony = False
        player.victoryPoints-=1
        player.nColonies-=1
        del player.ownedColonies[-1]

        if(place.harbor != ""):
            del player.ownedHarbors[-1]

def placeStreet(player, edge, undo = False, justCheck = False):
    previousLongestStreetOwner = player.game.longestStreetPlayer(justCheck)
    if(not undo):
        player.useResource("wood")
        player.useResource("clay")
        Board.Board().edges[edge] = player.id
        player.nStreets+=1
        player.ownedStreets.append(edge)
    else:
        Bank.Bank().giveResource(player, "wood")   
        Bank.Bank().giveResource(player, "clay")    
        Board.Board().edges[edge] = 0
        player.nStreets-=1
        del player.ownedStreets[-1]
    actualLongestStreetOwner = player.game.longestStreetPlayer(justCheck)

    if(previousLongestStreetOwner.id != actualLongestStreetOwner.id): 
        player.game.longestStreetOwner = actualLongestStreetOwner
        actualLongestStreetOwner.victoryPoints += 2
        #print("-2 riga 76. JustCheck(?) ", justCheck)
        previousLongestStreetOwner.victoryPoints -= 2

def placeColony(player, place: cg.Place, undo = False, justCheck = False):
    previousLongestStreetOwner = player.game.longestStreetPlayer(justCheck)
    if(not undo):
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

    else:
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

    actualLongestStreetOwner = player.game.longestStreetPlayer(justCheck)
    if(previousLongestStreetOwner.id != actualLongestStreetOwner.id):
        player.game.longestStreetOwner = actualLongestStreetOwner
        actualLongestStreetOwner.victoryPoints += 2
        previousLongestStreetOwner.victoryPoints -= 2
        

def placeCity(player, place: cg.Place, undo = False):
    if(not undo):
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

    else:
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

def buyDevCard(player, card, undo = False):
    if(not undo):
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

    else:
        print("Debug: BUG wrong way. (riga 114 move")
        Bank.Bank().giveResource(player, "iron")
        Bank.Bank().giveResource(player, "crop")
        Bank.Bank().giveResource(player, "sheep")

def discardResource(player, resource, undo = False):
    if(not undo):
        # print("debug riga 124 move")
        player.useResource(resource)
    else:
        # print("debug riga 124 move, undo")
        Bank.Bank().giveResource(player, resource)

def passTurn(player, temp=None):
    pass

def useRobber(player, tilePosition, undo = False, justCheck = False):
    if(not justCheck):
        stealResource(player, Board.Board().tiles[tilePosition])
    previousPosition = Board.Board().robberTile
    Board.Board().robberTile = tilePosition
    return previousPosition

def useKnight(player, tilePosition, undo = False, justCheck = False):
    largestArmy = player.game.largestArmy(justCheck)   

    previousPosition = Board.Board().robberTile
    Board.Board().robberTile = tilePosition
    if(not justCheck):
        stealResource(player, Board.Board().tiles[tilePosition])
    if(not undo):
        player.unusedKnights -= 1
        player.usedKnights += 1
    else:
        player.unusedKnights += 1
        player.usedKnights -= 1

    postMoveLargArmy = player.game.largestArmy(justCheck)

    if(largestArmy != postMoveLargArmy):
        postMoveLargArmy.victoryPoints += 2 
        #print("-2 riga 207")
        largestArmy.victoryPoints -= 2
    return previousPosition

def stealResource(player, tile: cg.Tile):
    playersInTile = []
    for place in tile.associatedPlaces:
        owner = player.game.players[Board.Board().places[place].owner-1]
        if owner not in playersInTile and owner.id != 0 and owner != player and owner.resourceCount() > 0: 
            playersInTile.append(owner)
    if len(playersInTile) > 0:
        chosenPlayer = playersInTile[random.randint(0,len(playersInTile)-1)]
        chosenPlayer.stealFromMe(player)
    return

def tradeBank(player, coupleOfResources, undo = False):
    toTake, toGive = coupleOfResources
    if(not undo):
        Bank.Bank().giveResource(player, toTake)
        for _ in range(0, Bank.Bank().resourceToAsk(player, toGive)):
            player.useResource(toGive)
        #print("TAKEN 1", toTake, " GIVEN ", Bank.Bank().resourceToAsk(player, toGive), toGive)
    else:
        player.useResource(toTake)
        for i in range(0, Bank.Bank().resourceToAsk(player, toGive)):
            Bank.Bank().giveResource(player, toGive)
        #print("UNDO OF: TAKEN 1", toTake, " GIVEN ", Bank.Bank().resourceToAsk(player, toGive), toGive)

def useMonopolyCard(player, resource):
    player.monopolyCard -= 1
    sum = 0
    for p in player.game.players:
        sum = sum + p.resources[resource]
        p.resources[resource] = 0
    player.resources[resource] = sum
    return
    
def useRoadBuildingCard(player, edges, undo = False, justCheck = False):
    assert len(edges) == 2, " FATAL ERROR. RoadBuildingCard: Number of elements can't be lower then 2, the edges must be passed in a tuple or list."
    if(not undo):
        player.roadBuildingCard -= 1
    else:
        player.roadBuildingCard += 1
    e1, e2 = edges
    if e1 is not None:
        placeFreeStreet(player, e1, undo, justCheck)
        print("User: ", player.id, ", Road building card used - first ", e1)
    if e2 is not None:
        placeFreeStreet(player, e2, undo, justCheck)
        print("User: ", player.id, "Road building card used - second ", e2)


def useYearOfPlentyCard(player, resources, undo = False):
    if not undo:
        player.yearOfPlentyCard -= 1
        Bank.Bank().giveResource(player, resources[0])
        Bank.Bank().giveResource(player, resources[1])
    else:
        player.yearOfPlentyCard += 1
        player.useResource(resources[0])
        player.useResource(resources[1])

def cardMoves():
    return [useMonopolyCard, useKnight, useYearOfPlentyCard, useRoadBuildingCard]

# import Board
# import Bank
# import Player
# import CatanGraph as cg