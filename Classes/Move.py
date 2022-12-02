import Bank, Player
from Board import Board
from CatanGraph import Place

#class Move:
def placeFreeStreet(player, edge, undo = False):
    if(not undo):
        Board().edges[edge] = player.id
        #print("debug: riga 9 in move ", str(edge[0]), "-", str(edge[1]))
    else:
        Board().edges[edge] = 0

#@staticmethod
def placeFreeColony(player: Player, place: Place, undo = False):
    if(not undo):
        Board().places[place.id].owner = player.id
        Board().places[place.id].isColony = True
        player.victoryPoints+=1
    else:
        Board().places[place.id].owner = 0
        Board().places[place.id].isColony = False
        player.victoryPoints-=1

def placeStreet(player, edge, undo = False):
    if(not undo):
        player.resources["wood"] = player.resources["wood"] -1
        player.resources["clay"] = player.resources["clay"] -1
        Bank.resources["wood"] = Bank.resources["wood"] + 1
        Bank.resources["clay"] = Bank.resources["clay"] + 1
        Board().edges[edge] = player.id
    else:
        player.resources["wood"] = player.resources["wood"] +1
        player.resources["clay"] = player.resources["clay"] +1
        Bank.resources["wood"] = Bank.resources["wood"] - 1
        Bank.resources["clay"] = Bank.resources["clay"] - 1
        Board().edges[edge] = 0

def placeColony(player, place, undo = False):
    if(not undo):
        player.resources["wood"] = player.resources["wood"] -1
        player.resources["clay"] = player.resources["clay"] -1
        player.resources["crop"] = player.resources["crop"] -1
        player.resources["sheep"] = player.resources["sheep"] -1
        Bank.resources["wood"] = Bank.resources["wood"] + 1
        Bank.resources["clay"] = Bank.resources["clay"] + 1
        Bank.resources["crop"] = Bank.resources["crop"] + 1
        Bank.resources["sheep"] = Bank.resources["sheep"] + 1
        Board().places[place].owner = player.id
        Board().place[place].isColony = True
        player.victoryPoints+=1
    else:
        player.resources["wood"] = player.resources["wood"] +1
        player.resources["clay"] = player.resources["clay"] +1
        player.resources["crop"] = player.resources["crop"] +1
        player.resources["sheep"] = player.resources["sheep"] +1
        Bank.resources["wood"] = Bank.resources["wood"] - 1
        Bank.resources["clay"] = Bank.resources["clay"] - 1
        Bank.resources["crop"] = Bank.resources["crop"] - 1
        Bank.resources["sheep"] = Bank.resources["sheep"] - 1
        Board().places[place].owner = 0
        Board().place[place].isColony = False
        player.victoryPoints-=1

def placeCity(player, place, undo = False):
    if(not undo):
        player.resources["iron"] = player.resources["iron"] -3
        player.resources["crop"] = player.resources["crop"] -2
        Bank.resources["iron"] = Bank.resources["iron"] +3
        Bank.resources["crop"] = Bank.resources["crop"] +2
        Board().places[place].isColony = False
        Board().places[place].isCity = True
        player.victoryPoints+=1
    else:
        player.resources["iron"] = player.resources["iron"] +3
        player.resources["crop"] = player.resources["crop"] +2
        Bank.resources["iron"] = Bank.resources["iron"] -3
        Bank.resources["crop"] = Bank.resources["crop"] -2
        Board().places[place].isColony = True
        Board().places[place].isCity = False
        player.victoryPoints-=1

def buyDevCard(player):
    player.resources["iron"] = player.resources["iron"] -1
    player.resources["crop"] = player.resources["crop"] -1
    player.resources["sheep"] = player.resources["sheep"] -1
    card = Board().deck[0]
    if(card == "knight"):
        player.justBoughtKnight = player.justBoughtKnight +1
    if(card == "monopoly"):
        player.justBoughtMonopolyCard = player.justBoughtMonopolyCard +1
    if(card == "road_building"):
        player.justBoughtRoadBuildingCard = player.justBoughtRoadBuildingCard +1
    if(card == "year_of_plenty"):
        player.justBoughtYearOfPlentyCard = player.justBoughtYearOfPlentyCard +1
    if(card == "victory_point"):
        player.victoryPoints = player.victoryPoints + 1
    return Board().deck[:1]

def discardResource(player, resource, undo = False):
    if(not undo):
        player.resources[resource] -= 1
        Bank().resources[resource] += 1
    else:
        player.resources[resource] += 1
        Bank().resources[resource] -= 1

def passTurn(player, temp):
    #player.game.nextTurn(player)
    pass

def useRobber(player, tilePosition, undo = False):
    previousPosition = Board().robberTile
    Board().robberTile = tilePosition
    return previousPosition

def useKnight(player, tilePosition, undo = False):
    largArmy = player.game.largestArmy()

    previousPosition = Board().robberTile
    Board().robberTile = tilePosition

    if(not undo):
        player.knightUsed += 1
    else:
        player.knightUsed -= 1

    postMoveLargArmy = player.game.largestArmy()

    if(largArmy != postMoveLargArmy):
        postMoveLargArmy.victoryPoints += 2 
        largArmy -= 2

    return previousPosition

def tradeBank(player, coupleOfResources, undo = False):
    if(not undo):
        toTake, toGive = coupleOfResources
        player.resources[toTake] = player.resources[toTake] + 1
        player.resources[toGive] = player.resources[toGive] - Bank.resourceToAsk(player, toGive)
    else:
        toTake, toGive = coupleOfResources
        player.resources[toTake] = player.resources[toTake] - 1
        player.resources[toGive] = player.resources[toGive] + Bank.resourceToAsk(player, toGive)

def useMonopolyCard(player, resource):
    player.monopolyCard = player.monopolyCard - 1
    sum = 0
    for p in player.game.players:
        sum = sum + p.resources[resource]
        p.resources[resource] = 0
    player.resources[resource] = sum
    return
    
def useRoadBuildingCard(player, edges, undo = False):
    placeStreet(edges[0], undo)
    placeStreet(edges[1], undo)

def useYearOfPlentyCard(player, resources, undo = False):
    if not undo:
        player.resources[resources[0]] +1
        player.resources[resources[0]] +1
    else:
        player.resources[resources[0]] -1
        player.resources[resources[0]] -1

def cardMoves():
    return [useMonopolyCard, useKnight, useYearOfPlentyCard, useRoadBuildingCard]

        