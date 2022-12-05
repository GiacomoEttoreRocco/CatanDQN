import Move
from Bank import Bank
from Board import Board
import random

class Player: 
    def __init__(self, id, game):

        self.debugColonies = []

        self.id = id
        self.victoryPoints = 0
        self.victoryPointsCards = 0

        self.game = game

        self.nColonies = 0
        self.nCities = 0
        self.nStreets = 0

        self.usedKnights = 0
        self.unusedKnights = 0
        self.justBoughtKnights = 0

        self.monopolyCard = 0
        self.justBoughtMonopolyCard = 0

        self.roadBuildingCard = 0
        self.justBoughtRoadBuildingCard = 0
        
        self.yearOfPlentyCard = 0
        self.justBoughtYearOfPlentyCard = 0

        # RESOURCES:
        self.resources = {"wood" : 0, "clay" : 0, "crop": 0, "sheep": 0, "iron": 0}

        #HARBORS: 
        self.ownedHarbors = []

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return self.id != other.id

    def __lt__(self, other):
        return self.id < other.id

    def __le__(self, other):
        return self.id <= other.id
        
    def useResource(self, resource):
        if(self.resources[resource] < 0):
            "FATAL ERROR. You should not be able to use this method."
        self.resources[resource] -= 1
        Bank().resources[resource] += 1

    def printStats(self):
        print("ID:  ", self.id," ", self.resources, ".",
            "\nIt has :" ,self.victoryPoints, " points. \
             \nNumber of cities: ", self.nCities, \
            "\nNumber of colonies: ", self.nColonies, \
            "\nNumber of streets: ", self.nStreets, \
            "\nNumber of used knights: ", self.usedKnights, \
            "\nNumber of unused knights: ", self.unusedKnights, \
            "\nNumber of just bought knights: ", self.justBoughtKnights, \
            "\nNumber of VP card: ", self.victoryPointsCards, \
            "\nBank resources:", Bank().resources,
            "\nOwned colonies: ", self.debugColonies)


    def printResources(self):
         print("Print resources of player:  ", self.id," ", self.resources, "\n.")
        

    def availableMoves(self, turnCardUsed):

        availableMoves = [Move.passTurn]

        if(self.resources["crop"] >= 1 and self.resources["iron"] >= 1 and self.resources["sheep"] >= 1):
            availableMoves.append(Move.buyDevCard)

        if(self.resources["wood"] >= 1 and self.resources["clay"] >= 1 and self.calculatePossibleColony() == []): # TEMPORANEAMENTE
            availableMoves.append(Move.placeStreet)

        if(self.resources["wood"] >= 1  and self.resources["clay"] >= 1 and self.resources["sheep"] >= 1 and self.resources["crop"] >= 1):
            availableMoves.append(Move.placeColony)

        if(self.resources["iron"] >= 3 and self.resources["crop"] >= 2):
            availableMoves.append(Move.placeCity)

        canTrade = False
        for resource in self.resources.keys():
            if(Bank().resourceToAsk(self, resource) <= self.resources[resource]):
                canTrade = True
        if(canTrade):
                availableMoves.append(Move.tradeBank)

        if(self.unusedKnights >= 1 and not turnCardUsed):
            availableMoves.append(Move.useKnight)    

        if(self.monopolyCard >= 1 and not turnCardUsed):
            availableMoves.append(Move.useMonopolyCard)

        if(self.roadBuildingCard >= 1 and not turnCardUsed):
            availableMoves.append(Move.useRoadBuildingCard)

        if(self.yearOfPlentyCard >= 1 and not turnCardUsed):
            availableMoves.append(Move.useYearOfPlentyCard)

        return availableMoves

    def connectedEmptyEdges(self, edge):
        p1 = edge[0]
        p2 = edge[1]
        toRet = []
        if(Board().places[p1].owner == 0 or Board().places[p1].owner == self.id):
            for p in Board().graph.listOfAdj[p1]:
                if(p2 != p):
                    edge = tuple(sorted([p1, p]))
                    if(Board().edges[edge] == 0):
                        toRet.append(edge)

        if(Board().places[p2].owner == 0 or Board().places[p2].owner == self.id):
            for p in Board().graph.listOfAdj[p2]:
                if(p1 != p):
                    edge = tuple(sorted([p2, p]))
                    if(Board().edges[edge] == 0):
                        toRet.append(edge)
        return toRet

    def calculatePossibleEdges(self):
        possibleEdges = []
        for edge in Board().edges.keys():
            if(Board().edges[edge] == self.id):
                #print(edge)
                if(self.connectedEmptyEdges(edge) != None):
                    possibleEdges.extend(self.connectedEmptyEdges(edge))
        return possibleEdges

    def calculatePossibleInitialColony(self):
        toRet = []
        for p in Board().places:
            if(p.owner == 0):
                available = True
                for padj in Board().graph.listOfAdj[p.id]:
                    if(Board().places[padj].owner != 0):
                        available = False
                if(available):
                    toRet.append(p)
        return toRet

    def calculatePossibleInitialStreets(self):
        #toRet = []
        for p in Board().places:
            if(p.owner == self.id):
                streetOccupied = False
                toRet = []
                for padj in Board().graph.listOfAdj[p.id]:
                    edge = tuple(sorted([p.id, padj]))
                    if(Board().edges[edge] != 0):
                        streetOccupied = True
                    toRet.append(edge)

                if(not streetOccupied):
                    return toRet


    def calculatePossibleColony(self):
        possibleColonies = []
        for p in Board().places:
            if(p.owner == 0):
                for p_adj in Board().graph.listOfAdj[p.id]:
                    edge = tuple(sorted([p.id, p_adj]))
                    if(Board().edges[edge] == self.id): #controlliamo che l'arco appartenga al giocatore, edges è un dictionary che prende in input l'edge e torna l'owner (il peso)
                        available = True
                        for p_adj_adj in Board().graph.listOfAdj[p_adj]:
                            if(Board().places[p_adj_adj].owner != 0):
                                available = False
                        if(available and Board().places[p_adj].owner == 0): # soluzione temporanea
                            possibleColonies.append(Board().places[p_adj])
        print("POSSIBLE COLONIES: ", possibleColonies)
        return possibleColonies

    def calculatePossibleCity(self):
        possibleCities = []
        for p in Board().places:
            if(p.owner == self.id and p.isColony == 1):
                possibleCities.append(p.id)
        return possibleCities

    def calculatePossibleTrades(self):
        possibleTrades = []
        for resource in self.resources.keys():
            if(self.resources[resource] >= Bank().resourceToAsk(self, resource)):
                for resourceToTake in self.resources.keys():
                    if(resourceToTake != resource):
                        possibleTrades.append((resourceToTake, resource))
        return possibleTrades
    
    def evaluate(self, move):
        if(move == Move.discardResource):
            possibleCards = [r for r in self.resources.keys() if self.resources[r] > 0]
            candidateCard = None
            max = 0
            for card in possibleCards:
                valutation = self.moveValue(move, card)
                if(max < valutation):
                    max = valutation
                    candidateCard = card
            return max, candidateCard
       
        if(move == Move.placeFreeStreet):
            possibleEdges = self.calculatePossibleInitialStreets()
            candidateEdge = None
            max = 0
            for edge in possibleEdges: 
                valutation = self.moveValue(move, edge)
                if(max < valutation):
                    max = valutation
                    candidateEdge = edge
            return max, candidateEdge
        if(move == Move.placeFreeColony):
            possibleColony = self.calculatePossibleInitialColony()
            #print(possibleColony)
            candidateColony = None
            max = 0
            for colony in possibleColony:
                valutation = self.moveValue(move, colony)
                if(max < valutation):
                    max = valutation
                    candidateColony = colony
            return max, candidateColony     

        if(move == Move.placeStreet):
            possibleEdges = self.calculatePossibleEdges()
            candidateEdge = None
            max = 0
            for edge in possibleEdges: 
                valutation = self.moveValue(move, edge)
                if(max < valutation):
                    max = valutation
                    candidateEdge = edge
            return max, candidateEdge
        
        if(move == Move.placeColony):
            possibleColony = self.calculatePossibleColony()
            candidateColony = None
            max = 0
            for colony in possibleColony:
                valutation = self.moveValue(move, colony)
                if(max < valutation):
                    max = valutation
                    candidateColony = colony
            return max, candidateColony

        if(move == Move.placeCity):
            possibleCity = self.calculatePossibleCity()
            candidateCity = None
            max = 0
            for city in possibleCity:
                valutation = self.moveValue(move, city)
                if(max < valutation):
                    max = valutation
                    candidateCity = city
            return max, candidateCity            

        if(move == Move.buyDevCard):
            valutation = self.moveValue(move, None)
            return valutation, None

        if(move == Move.passTurn):
            return self.moveValue(move), None

        if(move == Move.useKnight):
            max = 0
            for tile in Board().tiles: 
                valutation = self.moveValue(move, tile.identificator)
                if(max < valutation):
                    max = valutation
                    candidatePos = tile.identificator
            return max, candidatePos

        if(move == Move.useRobber):
            max = 0
            for tile in Board().tiles: 
                valutation = self.moveValue(move, tile.identificator)
                if(max < valutation):
                    max = valutation
                    candidatePos = tile.identificator
            return max, candidatePos        

        if(move == Move.tradeBank):
            possibleTrades = self.calculatePossibleTrades()
            candidateTrade = None
            max = 0
            for trade in possibleTrades:
                valutation = self.moveValue(move, trade)
                if(max < valutation):
                    max = valutation
                    candidateTrade = trade
            return max, candidateTrade

        if(move == Move.useMonopolyCard):
            max = 0
            for res in Bank().resources.keys():
                valutation = self.moveValue(move, res)
                if(max < valutation):
                    max = valutation
                    candidateRes = res
            return max, candidateRes

        if(move == Move.useYearOfPlentyCard):
            candidateRes = ()
            max = 0
            for res1 in Bank().resources.keys():
                for res2 in Bank().resources.key():
                    valutation = self.moveValue(move, (res1, res2))
                    if(max < valutation):
                        max = valutation
                        candidateRes = (res1, res2)
            return max, candidateRes

        if(move == Move.useRoadBuildingCard):
            possibleEdges = self.calculatePossibleEdges()
            candidateEdge1 = None
            max1 = 0
            for edge in possibleEdges: 
                valutation = self.moveValue(Move.placeFreeStreet, edge)
                if(max1 < valutation):
                    max1 = valutation
                    candidateEdge1 = edge

            possibleEdges = self.calculatePossibleEdges()
            candidateEdge2 = None
            max2 = 0
            for edge in possibleEdges: 
                if(edge != candidateEdge1):
                    valutation = self.moveValue(Move.placeFreeStreet, edge)
                    if(max2 < valutation):
                        max2 = valutation
                        candidateEdge2 = edge
            return max1+max2, [candidateEdge1, candidateEdge2]

    def totalCards(self):
        return sum(self.resources.values())

    def moveValue(self, move, thingNeeded = None):
        if(move == Move.passTurn):
            return 0.2 + random.uniform(0, 1)

        if(move == Move.useKnight):
            previousTilePos = move(self, thingNeeded, False, True)
            toRet = 1.5
            move(self, previousTilePos, True, True) # ATTUALMENTE è INUTILE SIA QUESTA RIGA CHE QUELLA SOPRA
            return toRet + random.uniform(0,2)

        if(move == Move.useRobber):
            previousTilePos = move(self, thingNeeded)
            toRet = 1.5
            move(self, previousTilePos, True) # ATTUALMENTE è INUTILE SIA QUESTA RIGA CHE QUELLA SOPRA
            return toRet + random.uniform(0,2)

        if(move == Move.buyDevCard):
            toRet = 1.5
            return toRet + random.uniform(0,5)

        if(move == Move.useMonopolyCard):
            toRet = 100.0
            return toRet

        move(self, thingNeeded)
        #toRet = 0
        if(move == Move.placeFreeColony):
            toRet = 10.0
        if(move == Move.placeFreeStreet):
            toRet = 10.0
        if(move == Move.placeCity):
            toRet = 1000
        if(move == Move.placeColony):
            toRet = 900
        if(move == Move.placeStreet and self.calculatePossibleColony() != None):
            toRet = 0.0
        if(move == Move.placeStreet and self.calculatePossibleColony() == None):
            toRet = 1990
        if(move == Move.buyDevCard):
            toRet = 1.0
        if(move == Move.tradeBank):
            toRet = 20.0
        if(move == Move.useRoadBuildingCard):
            toRet = 2.0
        if(move == Move.useYearOfPlentyCard):
            toRet = 2.0
        if(move == Move.discardResource):
            toRet = 1.0 

        #print("VALUE OF THE BOARD: ", toRet)

        move(self, thingNeeded, undo=True)

        return toRet + random.uniform(0,2)


