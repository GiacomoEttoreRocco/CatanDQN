import Command.commands as commands
import Command.controller as controller
import Classes.Bank as Bank
import Classes.Board as Board
from Classes.PlayerTypes import PlayerTypes
import random
import AI.Gnn as Gnn

class Player: 
    def __init__(self, id, game, type: PlayerTypes = PlayerTypes.NOT_SET):
         
        self.id = id
        self.type = type
        self.game = game

        self.ownedColonies = []
        self.ownedStreets = []
        self.ownedCities = []

        self.victoryPoints = 0
        self.victoryPointsCards = 0
        self.boughtCards = 0

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

        self.turnCardUsed = False

        self.lastRobberUser = False

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
        
    def chooseAction(self, actions):
        print("Mosse disponibili: ")
        for i, action in enumerate(actions):
            print("action ", i, ": ", action)
        if len(actions) == 1:     #Only possible action is passTurn
            toDo = 0
            print("Automatically passing turn...")
        else:
            toDo = int(input("Insert the index of the action you want to do: "))
            while(toDo >= len(actions)):
                toDo = int(input("Index too large. Try again: "))
        return actions[toDo][0], actions[toDo][1]

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

    def availableActions(self, turnCardUsed):
        availableactions = [commands.PassTurnCommand]
        if(self.resources["crop"] >= 1 and self.resources["iron"] >= 1 and self.resources["sheep"] >= 1 and len(Board.Board().deck) > 0):
            availableactions.append(commands.BuyDevCardCommand)
        if(self.resources["wood"] >= 1 and self.resources["clay"] >= 1 and self.nStreets < 15 and self.calculatePossibleEdges() != None): 
            availableactions.append(commands.PlaceStreetCommand)
        if(self.resources["wood"] >= 1  and self.resources["clay"] >= 1 and self.resources["sheep"] >= 1 and self.resources["crop"] >= 1):
            availableactions.append(commands.PlaceColonyCommand)
        if(self.resources["iron"] >= 3 and self.resources["crop"] >= 2):
            availableactions.append(commands.PlaceCityCommand)
        canTrade = False
        for resource in self.resources.keys():
            if(Bank.Bank().resourceToAsk(self, resource) <= self.resources[resource]):
                canTrade = True
        if(canTrade):
                availableactions.append(commands.TradeBankCommand)
        if(self.unusedKnights >= 1 and not turnCardUsed):
            availableactions.append(commands.UseKnightCommand)    
        if(self.monopolyCard >= 1 and not turnCardUsed):
            availableactions.append(commands.UseMonopolyCardCommand)
        if(self.roadBuildingCard >= 1 and not turnCardUsed):
            availableactions.append(commands.UseRoadBuildingCardCommand)
        if(self.yearOfPlentyCard >= 1 and not turnCardUsed):
            availableactions.append(commands.UseYearOfPlentyCardCommand)
        return availableactions

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
        if(len(self.ownedStreets) == 15):
            return possibleEdges
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
                        if(available and Board.Board().places[p_adj].owner == 0 and self.nColonies < 5 and Board.Board().places[p_adj] not in possibleColonies): 
                            possibleColonies.append(Board.Board().places[p_adj])
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
    
    def evaluate(self, action):
        if(action == commands.DiscardResourceCommand):
            possibleCards = [r for r in self.resources.keys() if self.resources[r] > 0]
            candidateCard = None
            max = -1
            for card in possibleCards:
                valutation = self.actionValue(action, card)
                if(max < valutation):
                    max = valutation
                    candidateCard = card
            return max, candidateCard
       
        if(action == commands.PlaceStreetCommand or action == commands.PlaceFreeStreetCommand):
            possibleEdges = self.calculatePossibleEdges()
            candidateEdge = None
            max = -1
            for edge in possibleEdges: 
                valutation = self.actionValue(action, edge)
                if(max < valutation):
                    max = valutation
                    candidateEdge = edge
            return max, candidateEdge

        if(action == commands.PlaceInitialStreetCommand):
            possibleEdges = self.calculatePossibleInitialStreets()
            candidateEdge = None
            max = -1
            for edge in possibleEdges: 
                valutation = self.actionValue(action, edge)
                if(max < valutation):
                    max = valutation
                    candidateEdge = edge
            return max, candidateEdge

        if(action == commands.PlaceInitialColonyCommand or action == commands.FirstChoiseCommand or action == commands.SecondChoiseCommand or action == commands.PlaceSecondColonyCommand):
            possibleColony = self.calculatePossibleInitialColony()
            candidateColony = None
            max = -1
            for colony in possibleColony:
                valutation = self.actionValue(action, colony)
                if(max < valutation):
                    max = valutation
                    candidateColony = colony
            return max, candidateColony    

        if(action == commands.PlaceColonyCommand):
            possibleColony = self.calculatePossibleColony()
            candidateColony = None
            max = -1
            for colony in possibleColony:
                valutation = self.actionValue(action, colony)
                if(max < valutation):
                    max = valutation
                    candidateColony = colony
            return max, candidateColony

        if(action == commands.PlaceCityCommand):
            possibleCity = self.calculatePossibleCity()
            candidateCity = None
            max = -1
            for city in possibleCity:
                valutation = self.actionValue(action, city)
                if(max < valutation):
                    max = valutation
                    candidateCity = city
            return max, candidateCity            

        if(action == commands.BuyDevCardCommand):
            valutation = self.actionValue(action, None)
            return valutation, None

        if(action == commands.PassTurnCommand):
            return self.actionValue(action), None

        if(action == commands.UseKnightCommand):
            max = -1
            for tile in Board.Board().tiles: 
                if(tile.identificator != Board.Board().robberTile):
                    valutation = self.actionValue(action, tile.identificator)
                    if(max < valutation):
                        max = valutation
                        candidatePos = tile.identificator
            return max, candidatePos

        if(action == commands.UseRobberCommand): # Yes they are the same method, but must be differentiated becouse of the count of knights.
            max = -1
            for tile in Board.Board().tiles: 
                if(tile.identificator != Board.Board().robberTile):
                    valutation = self.actionValue(action, tile.identificator)
                    if(max < valutation):
                        max = valutation
                        candidatePos = tile.identificator
            return max, candidatePos        

        if(action == commands.TradeBankCommand):
            possibleTrades = self.calculatePossibleTrades()
            candidateTrade = None
            max = -1
            for trade in possibleTrades:
                valutation = self.actionValue(action, trade)
                if(max < valutation):
                    max = valutation
                    candidateTrade = trade
            return max, candidateTrade

        if(action == commands.UseMonopolyCardCommand):
            max = -1
            for res in Bank.Bank().resources.keys():
                valutation = self.actionValue(action, res)
                if(max < valutation):
                    max = valutation
                    candidateRes = res
            return max, candidateRes

        if(action == commands.UseYearOfPlentyCardCommand):
            candidateRes = ()
            max = -1
            for res1 in Bank.Bank().resources.keys():
                for res2 in Bank.Bank().resources.keys():
                    if(Bank.Bank().resources[res1] > 0 and Bank.Bank().resources[res2] > 0):
                        valutation = self.actionValue(action, (res1, res2))
                    else:
                        valutation = -1
                    if(max < valutation):
                        max = valutation
                        candidateRes = (res1, res2)
            return max, candidateRes
        if(action == commands.UseRoadBuildingCardCommand):
            candidateEdge1 = None
            candidateEdge2 = None
            toRet = 0
            if len(self.ownedStreets) < 14:
                possibleEdges = self.calculatePossibleEdges()
                max1 = -1
                for edge in possibleEdges: 
                    valutation = self.actionValue(commands.PlaceFreeStreetCommand, edge)
                    if(max1 < valutation):
                        max1 = valutation
                        candidateEdge1 = edge
                toRet += max1
            if len(self.ownedStreets) < 15:
                possibleEdges = self.calculatePossibleEdges()
                max2 = -1
                for edge in possibleEdges: 
                    if(edge != candidateEdge1):
                        valutation = self.actionValue(commands.PlaceFreeStreetCommand, edge)
                        if(max2 < valutation):
                            max2 = valutation
                            candidateEdge2 = edge
                toRet += max2
            return toRet, [candidateEdge1, candidateEdge2]

    def resourceCount(self):
        return sum(self.resources.values())

    def stealFromMe(self):
        resourcesOfPlayer = []
        for keyRes in self.resources.keys():
            resourcesOfPlayer.extend([keyRes] * self.resources[keyRes])
        assert(len(resourcesOfPlayer) > 0)
        randomTake = random.randint(0, len(resourcesOfPlayer)-1)
        resourceTaken = resourcesOfPlayer[randomTake]
        # print("Steal: ",resourceTaken, "from player ", self.id, "which has ", self.resources[resourceTaken])
        return resourceTaken

    def actionValue(self, action, thingNeeded = None):
        assert self.type != PlayerTypes.NOT_SET
        if self.type == PlayerTypes.HYBRID:
            return self.hybridAiActionValue(action, thingNeeded)
        elif self.type == PlayerTypes.PRIORITY:
            return self.priorityActionValue(action, thingNeeded)
        elif self.type == PlayerTypes.PURE:
            return self.pureAiActionValue(action, thingNeeded)
        
    def priorityActionValue(self, action, thingNeeded = None):
        if(self.victoryPoints >= 8):
            ctr = controller.ActionController()
            ctr.execute(action(self, thingNeeded)) 
            if(self.victoryPoints >= 10):
                print("Priority Agent conclusive move! " + str(action))
                toRet = 300.0
                ctr.undo()
                return toRet
            ctr.undo()

        if(action == commands.PassTurnCommand):
            return 0.2 + random.uniform(0, 1)

        if(action == commands.UseKnightCommand):
            toRet = 1.5
            return toRet + random.uniform(0,2)

        if(action == commands.UseRobberCommand):
            toRet = 1.5
            return toRet + random.uniform(0,2)

        if(action == commands.BuyDevCardCommand):
            toRet = 1.5
            return toRet + random.uniform(0,5)

        if(action == commands.UseMonopolyCardCommand):
            toRet = 100.0
            return toRet
        if(action == commands.PlaceStreetCommand or action == commands.PlaceInitialStreetCommand or action == commands.PlaceColonyCommand):
            if(action == commands.PlaceColonyCommand):
                toRet = 90
            else:
                toRet = 16
            return toRet 
        if(action == commands.PlaceInitialColonyCommand):
            toRet = 10.0
        elif(action == commands.PlaceStreetCommand): 
            toRet = 10.0
        elif(action == commands.PlaceCityCommand):
            toRet = 100.0
        elif(action == commands.TradeBankCommand):
            toRet = 15.0
        elif(action == commands.UseRoadBuildingCardCommand):
            toRet = 2.0
        elif(action == commands.UseYearOfPlentyCardCommand):
            toRet = 200.0
        elif(action == commands.DiscardResourceCommand):
            toRet = 1.0 
        else:
            toRet = 0.5
        return toRet + random.uniform(0,2)

    def hybridAiActionValue(self, action, thingNeeded = None):
    # def HARDaiActionValue(self, action, thingNeeded = None):
        ctr = controller.ActionController()
        if(action == commands.FirstChoiseCommand or action == commands.SecondChoiseCommand or action == commands.PlaceInitialStreetCommand or action == commands.PlaceInitialColonyCommand or action == commands.PlaceSecondColonyCommand):
            # print("Initial path, player: ", self.id)
            ctr.execute(action(self, thingNeeded)) 
            toRet = Gnn.Gnn().evaluatePositionForPlayer(self)
            ctr.undo()
        else:
            if(action == commands.PassTurnCommand): 
                # print("pass turn path, player: ", self.id)
                toRet = Gnn.Gnn().evaluatePositionForPlayer(self)
            else:
                pointsBefore = self.victoryPoints
                previousPossibleColonies = self.calculatePossibleColony()
                previousCount = self.resourceCount()
                ctr.execute(action(self, thingNeeded)) 
                # print("Azione valutata: ", str(action))
                if(self.victoryPoints >= 10):
                    print("Hybrid Agent conclusive move! " + str(action))
                    ctr.undo()
                    return 1000.0
                if(pointsBefore < self.victoryPoints):
                    # print("Greedy path, player: ", self.id)
                    toRet = 300.0 + Gnn.Gnn().evaluatePositionForPlayer(self)
                elif(action == commands.PlaceStreetCommand or action == commands.PlaceFreeStreetCommand):
                    if(previousPossibleColonies == []):
                        if(self.calculatePossibleColony() != []):
                            # print("New colony available path, player: ", self.id)
                            # print("smart move")
                            val = Gnn.Gnn().evaluatePositionForPlayer(self)
                            toRet = 190.0 + val
                            # print("Valore val gnn:", val)
                        else:
                            # print("Street path, player: ", self.id)
                            # print("could be a smart move")
                            val = Gnn.Gnn().evaluatePositionForPlayer(self)
                            toRet = 185.0 + val
                            # print("Valore val gnn:", val)
                    elif(len(previousPossibleColonies) < len(self.calculatePossibleColony())):
                            # print("Another colony available path, player: ", self.id)
                            # print("Previous: ", previousPossibleColonies)
                            # print("Actual: ", self.calculatePossibleColony())
                            val = Gnn.Gnn().evaluatePositionForPlayer(self)
                            toRet = 170.0 + val
                            # print("Valore val gnn:", val)
                    else:
                        # print("Random street path, player: ", self.id)
                        toRet = Gnn.Gnn().evaluatePositionForPlayer(self)
                elif(action == commands.TradeBankCommand):
                    if(self.resources['crop'] > 1 and self.resources['iron'] > 2 and self.calculatePossibleCity() != []):
                        # print("City trade path, player: ", self.id)
                        toRet = 175.0 + Gnn.Gnn().evaluatePositionForPlayer(self)
                    elif(self.resources['crop'] > 0 and self.resources['iron'] > 0 and self.resources['sheep'] > 0):
                        # print("City trade path, player: ", self.id)
                        toRet = 175.0 + Gnn.Gnn().evaluatePositionForPlayer(self)
                    elif(self.resources['wood'] > 0 and self.resources['clay'] > 0 and self.resources['crop'] > 0 and self.resources['sheep'] > 0 and previousPossibleColonies != []):
                        # print("Colony trade path, player: ", self.id)
                        toRet = 175.0 + Gnn.Gnn().evaluatePositionForPlayer(self)
                    elif(self.resources['wood'] > 0 and self.resources['clay'] > 0 and previousCount >= 7):
                        # print("Wood and Clay greater then 0 path, player: ", self.id)
                        # print("thing needed: ", thingNeeded)
                        toRet = 175.0 + Gnn.Gnn().evaluatePositionForPlayer(self)
                    elif(previousCount >= 8):
                        # print("Avoid discard path, player: ", self.id)
                        toRet = 100.0 + Gnn.Gnn().evaluatePositionForPlayer(self)
                    else:
                        # print("Random trade path, player: ", self.id)
                        toRet = Gnn.Gnn().evaluatePositionForPlayer(self)
                else:
                    # print("Random path, player: ", self.id)
                    toRet = Gnn.Gnn().evaluatePositionForPlayer(self)
                ctr.undo()
        # print("Evalutation from GNN: ", toRet)
        return toRet 

    def pureAiActionValue(self, action, thingNeeded = None):
        ctr = controller.ActionController()
        if(action == commands.PassTurnCommand): 
            toRet = Gnn.Gnn().evaluatePositionForPlayer(self)
        else:
            previousCount = self.resourceCount()
            ctr.execute(action(self, thingNeeded)) 
            if(self.victoryPoints >= 10):
                print("PureAI Agent conclusive move! " + str(action))
                ctr.undo()
                return 1000.0
            elif(previousCount >= 8):
                toRet = 100.0 + Gnn.Gnn().evaluatePositionForPlayer(self)
                ctr.undo()
            else:
                toRet = Gnn.Gnn().evaluatePositionForPlayer(self)
                ctr.undo()
        return toRet # 


    def globalFeaturesToDict(self):
        # for p in self.game.players:
        #     print("Id: ", p.id, "LAST USER: ", int(p.lastRobberUser))
        return {'player_id': self.id,'victory_points': self.victoryPoints, 'cards_bought': self.boughtCards, 'last_robber_user': int(self.lastRobberUser),
                'used_knights': self.usedKnights, 'crop': self.resources["crop"], 'iron': self.resources["iron"],
                'wood': self.resources["wood"], 'clay': self.resources["clay"], 'sheep': self.resources["sheep"], 'winner':None}