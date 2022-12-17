
import Classes.Move as Move
import Classes.Bank as Bank
import Classes.Board as Board
import random
import AI.Gnn as Gnn

class Player: 
    def __init__(self, id, game, AI = False, RANDOM = False):
        assert (not  (AI and RANDOM), "Error in definition of player")
        self.AI = AI
        self.RANDOM = RANDOM

        self.ownedColonies = []
        self.ownedStreets = []
        self.ownedCities = []

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
        Bank.Bank().resources[resource] += 1

    def chooseMove(self, moves):
        print("Mosse disponibili: ")
        for i, move in enumerate(moves):
            print("Move ", i, ": ", move)
        if len(moves) == 1:     #Only possible move is passTurn
            toDo = 0
            print("Automatically passing turn...")
        else:
            toDo = int(input("Insert the index of the move you want to do: "))
            while(toDo >= len(moves)):
                toDo = int(input("Index too large. Try again: "))
        return moves[toDo][0], moves[toDo][1]

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
            "\nBank resources:", Bank.Bank().resources,
            "\nOwned colonies: ", self.ownedColonies,
            "\nOwned cities: ", self.ownedCities,
            "\nOwned streets: ", self.ownedStreets,

            "\nCONTROLLER OF THE LARGEST ARMY:  ", self.game.largestArmyPlayer.id,
            "\nCONTROLLER OF THE LONGEST STREET:  ", self.game.longestStreetOwner.id, " of length: ", self.game.longestStreetLength)


    def printResources(self):
         print("Print resources of player:  ", self.id," ", self.resources, "\n")

    def availableMoves(self, turnCardUsed):
        availableMoves = [Move.passTurn]
        if(self.resources["crop"] >= 1 and self.resources["iron"] >= 1 and self.resources["sheep"] >= 1 and len(Board.Board().deck) > 0):
            availableMoves.append(Move.buyDevCard)
        if(self.resources["wood"] >= 1 and self.resources["clay"] >= 1 and self.nStreets < 15 and self.calculatePossibleEdges() != None): 
            availableMoves.append(Move.placeStreet)
        if(self.resources["wood"] >= 1  and self.resources["clay"] >= 1 and self.resources["sheep"] >= 1 and self.resources["crop"] >= 1):
            availableMoves.append(Move.placeColony)
        if(self.resources["iron"] >= 3 and self.resources["crop"] >= 2):
            availableMoves.append(Move.placeCity)
        canTrade = False
        for resource in self.resources.keys():
            if(Bank.Bank().resourceToAsk(self, resource) <= self.resources[resource]):
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

    def availableMovesWithInput(self, turnCardUsed):
        availableMoves = [(Move.passTurn, None)]
        if(self.resources["crop"] >= 1 and self.resources["iron"] >= 1 and self.resources["sheep"] >= 1 and len(Board.Board().deck) > 0):
            availableMoves.append((Move.buyDevCard, None))
        if(self.resources["wood"] >= 1 and self.resources["clay"] >= 1 and self.calculatePossibleColony() == [] and self.nStreets < 15 and self.calculatePossibleEdges() != None): # TEMPORANEAMENTE
            for street in self.calculatePossibleEdges():
                availableMoves.append((Move.placeStreet, street))
        if(self.resources["wood"] >= 1  and self.resources["clay"] >= 1 and self.resources["sheep"] >= 1 and self.resources["crop"] >= 1):
            for colony in self.calculatePossibleColony():
                availableMoves.append((Move.placeColony, colony))
        if(self.resources["iron"] >= 3 and self.resources["crop"] >= 2):
            for city in self.calculatePossibleCity():
                availableMoves.append((Move.placeCity, city))
        for resource in self.resources.keys():
            if(Bank.Bank().resourceToAsk(self, resource) <= self.resources[resource]):
                for res in self.resources.keys():
                    if(resource != res):
                        availableMoves.append((Move.tradeBank, (res, resource)))
        if(self.unusedKnights >= 1 and not turnCardUsed):
            for i in range(0, 19):
                if(i != Board.Board().robberTile):
                    availableMoves.append((Move.useKnight, i))    
        if(self.monopolyCard >= 1 and not turnCardUsed):
            for res in self.resources.keys():
                availableMoves.append((Move.useMonopolyCard, res))
        if(self.roadBuildingCard >= 1 and not turnCardUsed):
            cpe = self.calculatePossibleEdges()
            for is1 in range(0, len(cpe)):
                for is2 in range(is1, len(cpe)):
                    street1 = cpe[is1]
                    street2 = cpe[is2]
                    if(street1 != street2):
                        availableMoves.append((Move.useRoadBuildingCard, (street1, street2)))
        if(self.yearOfPlentyCard >= 1 and not turnCardUsed):
            ress = list(self.resources.keys())
            for ires1 in range(0, len(ress)):
                for ires2 in range(ires1, len(ress)):
                    availableMoves.append((Move.useYearOfPlentyCard, (ress[ires1], ress[ires2])))
        return availableMoves

    def connectedEmptyEdges(self, edge):
        p1 = edge[0]
        p2 = edge[1]
        toRet = []
        if(Board.Board().places[p1].owner == 0 or Board.Board().places[p1].owner == self.id):
            for p in Board.Board().graph.listOfAdj[p1]:
                if(p2 != p):
                    edge = tuple(sorted([p1, p]))
                    if(Board.Board().edges[edge] == 0):
                        toRet.append(edge)

        if(Board.Board().places[p2].owner == 0 or Board.Board().places[p2].owner == self.id):
            for p in Board.Board().graph.listOfAdj[p2]:
                if(p1 != p):
                    edge = tuple(sorted([p2, p]))
                    if(Board.Board().edges[edge] == 0):
                        toRet.append(edge)
        return toRet

    def calculatePossibleEdges(self):
        possibleEdges = []
        for edge in Board.Board().edges.keys():
            if(Board.Board().edges[edge] == self.id):
                if(edge == None):
                    print(Board.Board().edges[edge])
                if(self.connectedEmptyEdges(edge) != None):
                    possibleEdges.extend(self.connectedEmptyEdges(edge))
        return possibleEdges

    def calculatePossibleInitialColony(self):
        toRet = []
        for p in Board.Board().places:
            if(p.owner == 0):
                available = True
                for padj in Board.Board().graph.listOfAdj[p.id]:
                    if(Board.Board().places[padj].owner != 0):
                        available = False
                if(available):
                    toRet.append(p)
        return toRet

    def calculatePossibleInitialStreets(self):
        for p in Board.Board().places:
            if(p.owner == self.id):
                streetOccupied = False
                toRet = []
                for padj in Board.Board().graph.listOfAdj[p.id]:
                    edge = tuple(sorted([p.id, padj]))
                    if(Board.Board().edges[edge] != 0):
                        streetOccupied = True
                    toRet.append(edge)
                if(not streetOccupied):
                    return toRet

    def calculatePossibleColony(self):
        possibleColonies = []
        for p in Board.Board().places:
            if(p.owner == 0):
                for p_adj in Board.Board().graph.listOfAdj[p.id]:
                    edge = tuple(sorted([p.id, p_adj]))
                    if(Board.Board().edges[edge] == self.id): #controlliamo che l'arco appartenga al giocatore, edges è un dictionary che prende in input l'edge e torna l'owner (il peso)
                        available = True
                        for p_adj_adj in Board.Board().graph.listOfAdj[p_adj]:
                            if(Board.Board().places[p_adj_adj].owner != 0):
                                available = False
                        if(available and Board.Board().places[p_adj].owner == 0 and self.nColonies < 5): # soluzione temporanea
                            possibleColonies.append(Board.Board().places[p_adj])
        #print("POSSIBLE COLONIES: ", possibleColonies)
        return possibleColonies

    def calculatePossibleCity(self):
        possibleCities = []
        for p in Board.Board().places:
            if(p.owner == self.id and p.isColony == 1 and self.nCities < 4):
                possibleCities.append(p)
        return possibleCities

    def calculatePossibleTrades(self):
        possibleTrades = []
        for resource in self.resources.keys():
            if(self.resources[resource] >= Bank.Bank().resourceToAsk(self, resource)):
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
            possibleEdges = self.calculatePossibleEdges()
            candidateEdge = None
            max = 0
            for edge in possibleEdges: 
                valutation = self.moveValue(move, edge)
                if(max < valutation):
                    max = valutation
                    candidateEdge = edge
            return max, candidateEdge

        if(move == Move.placeInitialStreet):
            possibleEdges = self.calculatePossibleInitialStreets()
            candidateEdge = None
            max = 0
            for edge in possibleEdges: 
                valutation = self.moveValue(move, edge)
                if(max < valutation):
                    max = valutation
                    candidateEdge = edge
            return max, candidateEdge

        if(move == Move.placeInitialColony):
            possibleColony = self.calculatePossibleInitialColony()
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
            for tile in Board.Board().tiles: 
                if(tile.identificator != Board.Board().robberTile):
                    valutation = self.moveValue(move, tile.identificator)
                    if(max < valutation):
                        max = valutation
                        candidatePos = tile.identificator
            return max, candidatePos

        if(move == Move.useRobber): # Yes they are the same method, but must be differentiated becouse of the count of knights.
            max = 0
            for tile in Board.Board().tiles: 
                if(tile.identificator != Board.Board().robberTile):
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
            for res in Bank.Bank().resources.keys():
                valutation = self.moveValue(move, res)
                if(max < valutation):
                    max = valutation
                    candidateRes = res
            return max, candidateRes

        if(move == Move.useYearOfPlentyCard):
            candidateRes = ()
            max = 0
            for res1 in Bank.Bank().resources.keys():
                for res2 in Bank.Bank().resources.keys():
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

    def resourceCount(self):
        return sum(self.resources.values())

    def stealFromMe(self, player):
        resourcesOfPlayer = []
        for keyRes in self.resources.keys():
            resourcesOfPlayer.extend([keyRes] * self.resources[keyRes])
        assert(len(resourcesOfPlayer) > 0)
        randomTake = random.randint(0, len(resourcesOfPlayer)-1)
        resourceTaken = resourcesOfPlayer[randomTake]
        # print("RESOURCE TAKEN ---------------------------------------------------->", resourceTaken)
        self.resources[resourceTaken] -= 1
        player.resources[resourceTaken] += 1

    def moveValue(self, move, thingNeeded = None):
        if self.AI:
            return self.aiMoveValue(move, thingNeeded)
        elif self.RANDOM:
            return self.randomMoveValue(move, thingNeeded)
        
    def randomMoveValue(self, move, thingNeeded = None):
        if(move == Move.passTurn):
            return 0.2 + random.uniform(0, 1)

        if(move == Move.useKnight):
            previousTilePos = move(self, thingNeeded, False, True)
            toRet = 1.5
            move(self, previousTilePos, True, True) 
            return toRet + random.uniform(0,2)

        if(move == Move.useRobber):
            previousTilePos = move(self, thingNeeded, False, True)
            toRet = 1.5
            move(self, previousTilePos, True, True) 
            return toRet + random.uniform(0,2)

        if(move == Move.buyDevCard):
            toRet = 1.5
            return toRet + random.uniform(0,5)

        if(move == Move.useMonopolyCard):
            toRet = 100.0
            return toRet

        if(move == Move.placeFreeStreet or move == Move.placeStreet or move == Move.placeInitialStreet):
            move(self, thingNeeded, False, True)
            toRet = 16
            move(self, thingNeeded, True, True) 
            return toRet + random.uniform(0,2)
        move(self, thingNeeded)
        if(move == Move.placeInitialColony):
            toRet = 10.0
        if(move == Move.placeFreeStreet):
            toRet = 10.0
        if(move == Move.placeCity):
            toRet = 100.0
        if(move == Move.placeColony):
            toRet = 90.0
        if(move == Move.tradeBank):
            toRet = 15.0
        if(move == Move.useRoadBuildingCard):
            toRet = 2.0
        if(move == Move.useYearOfPlentyCard):
            toRet = 2000.0
        if(move == Move.discardResource):
            toRet = 1.0 
        move(self, thingNeeded, undo=True)
        return toRet + random.uniform(0,2)

    def aiMoveValue(self, move, thingNeeded = None):
        if(move == Move.passTurn):
            toRet = Gnn.Gnn().evaluatePosition(self)
        elif(move == Move.useKnight):
            previousTilePos = move(self, thingNeeded, False, True)
            toRet = Gnn.Gnn().evaluatePosition(self)
            move(self, previousTilePos, True, True) 
            # return toRet
        elif(move == Move.useRobber):
            previousTilePos = move(self, thingNeeded, False, True)
            toRet = Gnn.Gnn().evaluatePosition(self)
            move(self, previousTilePos, True, True) 
            # return toRet
        elif(move == Move.buyDevCard):
            toRet = Gnn.Gnn().evaluatePosition(self)
            # return toRet + random.uniform(-0.1,0.1)
        elif(move == Move.useMonopolyCard):
            toRet = Gnn.Gnn().evaluatePosition(self)
            # return toRet + random.uniform(-0.1,0.1)
        elif(move == Move.placeFreeStreet or move == Move.placeStreet or move == Move.placeInitialStreet):
            move(self, thingNeeded, False, True)
            toRet = Gnn.Gnn().evaluatePosition(self)
            move(self, thingNeeded, True, True) 
            # return toRet
        else:
            move(self, thingNeeded)
            toRet = Gnn.Gnn().evaluatePosition(self)
            move(self, thingNeeded, undo=True)
        print("Move: ", move, " Object: " , thingNeeded, " Value: ", toRet, "Player: ", self.id)
        return toRet

    def globalFeaturesToDict(self):
        return {'player_id': self.id,'victory_points': self.victoryPoints,\
            'used_knights': self.usedKnights, 'crop': self.resources["crop"], 'iron': self.resources["iron"],\
            'wood': self.resources["wood"], 'clay': self.resources["clay"], 'sheep': self.resources["sheep"], 'winner':None}

