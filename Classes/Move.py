import Player
from Board import Board
from Bank import Bank
from CatanGraph import Place

def placeFreeStreet(player, edge, undo = False):
    if(not undo):
        Board().edges[edge] = player.id
        player.nStreets+=1
    else:
        Board().edges[edge] = 0
        player.nStreets-=1


def placeFreeColony(player: Player, place: Place, undo = False):
    if(not undo):
        Board().places[place.id].owner = player.id
        Board().places[place.id].isColony = True
        player.victoryPoints+=1
        player.nColonies+=1

    else:
        Board().places[place.id].owner = 0
        Board().places[place.id].isColony = False
        player.victoryPoints-=1
        player.nColonies-=1



def placeStreet(player, edge, undo = False):
    if(not undo):
        player.useResource("wood")
        player.useResource("clay")
        Board().edges[edge] = player.id
        player.nStreets+=1

    else:
        Bank().giveResource(player, "wood")   
        Bank().giveResource(player, "clay")    
        Board().edges[edge] = 0
        player.nStreets-=1

def placeColony(player, place, undo = False):
    if(not undo):
        player.useResource("wood")
        player.useResource("clay")
        player.useResource("crop")
        player.useResource("sheep")

        Board().places[place].owner = player.id
        Board().places[place].isColony = True

        player.victoryPoints+=1
        player.nColonies+=1

        if(Board().places[place].harbor != ""):
            player.ownedHarbors.append(Board().places[place].harbor)
    else:
        Bank().giveResource(player, "wood")   
        Bank().giveResource(player, "clay")  
        Bank().giveResource(player, "crop")   
        Bank().giveResource(player, "sheep")  

        Board().places[place].owner = 0
        Board().places[place].isColony = False
        player.victoryPoints-=1
        player.nColonies-=1
        if(Board().places[place].harbor != ""):
            del player.ownedHarbors[-1]


def placeCity(player, place, undo = False):
    if(not undo):
        player.useResource("iron")
        player.useResource("iron")
        player.useResource("iron")
        player.useResource("crop")
        player.useResource("crop")

        Board().places[place].isColony = False
        Board().places[place].isCity = True
        player.victoryPoints+=1
        player.nCities+=1
    else:
        Bank().giveResource(player, "iron")
        Bank().giveResource(player, "iron")
        Bank().giveResource(player, "iron")
        Bank().giveResource(player, "crop")
        Bank().giveResource(player, "crop")

        Board().places[place].isColony = True
        Board().places[place].isCity = False
        player.victoryPoints-=1
        player.nCities-=1

def buyDevCard(player, card, undo = False):
    if(not undo):
        player.useResource("iron")
        player.useResource("crop")
        player.useResource("sheep")
        card = Board().deck[0]
        if(card == "knight"):
            player.justBoughtKnights = player.justBoughtKnights +1
        if(card == "monopoly"):
            player.justBoughtMonopolyCard = player.justBoughtMonopolyCard +1
        if(card == "road_building"):
            player.justBoughtRoadBuildingCard = player.justBoughtRoadBuildingCard +1
        if(card == "year_of_plenty"):
            player.justBoughtYearOfPlentyCard = player.justBoughtYearOfPlentyCard +1
        if(card == "victory_point"):
            player.victoryPoints = player.victoryPoints + 1
        Board().deck = Board().deck[:1] 
    else:
        print("Debug: BUG wrong way. (riga 114 move")
        Bank().giveResource(player, "iron")
        Bank().giveResource(player, "crop")
        Bank().giveResource(player, "sheep")


def discardResource(player, resource, undo = False):
    if(not undo):
        print("debug riga 124 move")
        player.useResource(resource)
    else:
        print("debug riga 124 move, undo")
        Bank().giveResource(player, resource)

def passTurn(player, temp):
    pass

def useRobber(player, tilePosition, undo = False):
    previousPosition = Board().robberTile
    Board().robberTile = tilePosition
    return previousPosition

def useKnight(player, tilePosition, undo = False, justCheck = False):
    largArmy = player.game.largestArmy(justCheck)
    previousPosition = Board().robberTile
    Board().robberTile = tilePosition
    if(not undo):
        player.unusedKnights -= 1
        player.usedKnights += 1
    else:
        player.unusedKnights += 1
        player.usedKnights -= 1
    postMoveLargArmy = player.game.largestArmy(justCheck)
    if(largArmy != postMoveLargArmy):
        postMoveLargArmy.victoryPoints += 2 
        print("Debug +2 punti, largest riga 153, move")
        largArmy.victoryPoints -= 2
        print("Debug -2 punti, largest riga 153, move")
    return previousPosition

def tradeBank(player, coupleOfResources, undo = False):
    toTake, toGive = coupleOfResources
    if(not undo):
        player.resources[toTake] += 1
        player.resources[toGive] -= Bank().resourceToAsk(player, toGive)
        print("TAKEN 1", toTake, " GIVEN ", Bank().resourceToAsk(player, toGive), toGive)
    else:
        player.resources[toTake] -= 1
        player.resources[toGive] += Bank().resourceToAsk(player, toGive)
        print("UNDO OF: TAKEN 1", toTake, " GIVEN ", Bank().resourceToAsk(player, toGive), toGive)


def useMonopolyCard(player, resource):
    player.monopolyCard = player.monopolyCard - 1
    sum = 0
    for p in player.game.players:
        sum = sum + p.resources[resource]
        p.resources[resource] = 0
    player.resources[resource] = sum
    return
    
def useRoadBuildingCard(player, edges, undo = False):
    placeFreeStreet(player, edges[0], undo)
    placeFreeStreet(player, edges[1], undo)

def useYearOfPlentyCard(player, resources, undo = False):
    if not undo:
        Bank().giveResource(player, resources[0])
        Bank().giveResource(player, resources[1])
    else:
        player.useResource(resources[0])
        player.useResource(resources[1])

def cardMoves():
    return [useMonopolyCard, useKnight, useYearOfPlentyCard, useRoadBuildingCard]

        