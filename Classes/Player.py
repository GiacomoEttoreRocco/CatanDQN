from Classes.MoveTypes import TurnMoveTypes
from Classes.Strategy.Strategy import Strategy
from Classes.staticUtilities import *
import Command.commands as commands
import Command.controller as controller
import Classes.Bank as Bank
import Classes.Board as Board
# from Classes.PlayerTypes import PlayerTypes
import random
import AI.Gnn as Gnn
import torch

class Player: 
    def __init__(self, id, game, strategy : Strategy):
        self.id = id
        # if(strategy == None):
        self.strategy = strategy
        self.game = game
        self.ownedColonies = []
        self.ownedStreets = []
        self.ownedCities = []
        self._victoryPoints = 0
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
        self.resources = {"wood" : 0, "clay" : 0, "crop": 0, "sheep": 0, "iron": 0}
        self.ownedHarbors = []
        # for RL stuff

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return self.id != other.id

    def __lt__(self, other):
        return self.id < other.id

    def __le__(self, other):
        return self.id <= other.id
        
    # def humanChooseAction(self, actions):
    #     print("Mosse disponibili: ")
    #     for i, action in enumerate(actions):
    #         print("action ", i, ": ", action)
    #     if len(actions) == 1:     #Only possible action is passTurn
    #         toDo = 0
    #         print("Automatically passing turn...")
    #     else:
    #         toDo = int(input("Insert the index of the action you want to do: "))
    #         while(toDo >= len(actions)):
    #             toDo = int(input("Index too large. Try again: "))
    #     return actions[toDo][0], actions[toDo][1]

    def reset(self):
        self.ownedColonies = []
        self.ownedStreets = []
        self.ownedCities = []
        self._victoryPoints = 0
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
        self.resources = {"wood" : 0, "clay" : 0, "crop": 0, "sheep": 0, "iron": 0}
        self.ownedHarbors = []

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
        availableActions = [commands.PassTurnCommand]
        if(availableResourcesForDevCard(self.resources) and len(Board.Board().deck) > 0):
            availableActions.append(commands.BuyDevCardCommand)
        if(availableResourcesForStreet(self.resources) and self.nStreets < 15 and len(self.calculatePossibleStreets()) > 0): 
            availableActions.append(commands.PlaceStreetCommand)
        if(availableResourcesForColony(self.resources) and self.nColonies < 5 and len(self.calculatePossibleColonies()) > 0):
            availableActions.append(commands.PlaceColonyCommand)
        if(availableResourcesForCity(self.resources) and self.nCities < 4 and len(self.calculatePossibleCities()) > 0):
            availableActions.append(commands.PlaceCityCommand)
        canTrade = False
        for resource in self.resources.keys():
            if(Bank.Bank().resourceToAsk(self, resource) <= self.resources[resource]):
                canTrade = True
        if(canTrade):
                availableActions.append(commands.TradeBankCommand)
        if(self.unusedKnights >= 1 and not turnCardUsed):
            availableActions.append(commands.UseKnightCommand)    
        if(self.monopolyCard >= 1 and not turnCardUsed):
            availableActions.append(commands.UseMonopolyCardCommand)
        if(self.roadBuildingCard >= 1 and not turnCardUsed and self.nStreets < 14 and len(self.calculatePossibleStreets()) >= 2):
            availableActions.append(commands.UseRoadBuildingCardCommand)
        if(self.yearOfPlentyCard >= 1 and not turnCardUsed):
            availableActions.append(commands.UseYearOfPlentyCardCommand)
        return availableActions
    
    def availableTurnActionsId(self):
        availableActions = [TurnMoveTypes.PassTurn.value] 
        if(availableResourcesForDevCard(self.resources) and len(Board.Board().deck) > 0):
            availableActions.append(TurnMoveTypes.BuyDevCard.value) 
        if(availableResourcesForStreet(self.resources) and self.nStreets < 15 and len(self.calculatePossibleStreets()) > 0): 
            availableActions.append(TurnMoveTypes.PlaceStreet.value) 
        if(availableResourcesForColony(self.resources) and self.nColonies < 5 and len(self.calculatePossibleColonies()) > 0):
            availableActions.append(TurnMoveTypes.PlaceColony.value) 
        if(availableResourcesForCity(self.resources) and self.nCities < 4 and len(self.calculatePossibleCities()) > 0):
            availableActions.append(TurnMoveTypes.PlaceCity.value)
        canTrade = False
        for resource in self.resources.keys():
            if(Bank.Bank().resourceToAsk(self, resource) <= self.resources[resource]):
                canTrade = True
        if(canTrade):
                availableActions.append(TurnMoveTypes.TradeBank.value) 
        if(self.unusedKnights >= 1 and not self.turnCardUsed):
            availableActions.append(TurnMoveTypes.UseKnight.value)
        if(self.monopolyCard >= 1 and not self.turnCardUsed):
            availableActions.append(TurnMoveTypes.UseMonopolyCard.value) 
        if(self.roadBuildingCard >= 1 and not self.turnCardUsed and self.nStreets < 14 and len(self.calculatePossibleStreets()) >= 2):
            availableActions.append(TurnMoveTypes.UseRoadBuildingCard.value)
        if(self.yearOfPlentyCard >= 1 and not self.turnCardUsed):
            availableActions.append(TurnMoveTypes.UseYearOfPlentyCard.value)
        return availableActions

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

    def calculatePossibleStreets(self):
        possibleEdges = []
        if(len(self.ownedStreets) == 15):
            return possibleEdges
        for edge in Board.Board().edges.keys():
            if(Board.Board().edges[edge] == self.id):
                if(edge == None):
                    print("Debug, FATAL ERROR. Riga 192 player: ", Board.Board().edges[edge]) # debug
                if(self.connectedEmptyEdges(edge) != None):
                    possibleEdges.extend(self.connectedEmptyEdges(edge))
        return possibleEdges

    def calculatePossibleInitialColonies(self):
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

    def calculatePossibleColonies(self):
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

    def calculatePossibleCities(self):
        possibleCities = []
        for p in Board.Board().places:
            if(p.owner == self.id and p.isColony == 1 and self.nCities < 4):
                possibleCities.append(p)
        return possibleCities

    def calculatePossibleTrades(self):
        possibleTrades = []
        for resource in self.resources.keys():
            if(self.resources[resource] >= Bank.Bank().resourceToAsk(self, resource)):
                # print("Riga 246 player: ", self.resources[resource], ">=", Bank.Bank().resourceToAsk(self, resource))
                for resourceToTake in self.resources.keys():
                    if(resourceToTake != resource):
                        possibleTrades.append((resourceToTake, resource))
        # print("########################################")
        # print(possibleTrades)
        # print("########################################")
        return possibleTrades
    
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
    
    def globalStateTensor(self):
        myCrop = self.resources["crop"]
        myIron = self.resources["iron"]
        myWood = self.resources["wood"]
        myClay = self.resources["clay"]
        mySheep = self.resources["sheep"]
        totResOthers = Bank.Bank().totalResourceOut() - (myCrop + myIron + myWood + myClay + mySheep)
        tensor = torch.tensor([[self._victoryPoints, self.boughtCards, self.usedKnights, myCrop, myIron, myWood, myClay, mySheep, totResOthers]], dtype=torch.float)
        return tensor
    
    def bestAction(self):
        return self.strategy.bestAction(self) 
    
    def victoryPointsModification(self, points):
        self._victoryPoints += points

    def isLeaf(self, place): #  Per il Jack del futuro: non è errato, richiede un po' di ragionamento
        streetCounter = 0
        for street in self.ownedStreets:
            if(place in street):
                streetCounter+=1
        if(streetCounter == 1):
            return True
        return False
    
    def isTriple(self, place):
        streetCounter = 0
        for street in self.ownedStreets:
            if(place in street):
                streetCounter+=1
        if(streetCounter == 3):
            return True
    
    def findStartingPoints(self):
        startingPoints = []
        for edge in self.ownedStreets:
            p1, p2 = edge
            if (self.isLeaf(p1) or self.isTriple(p1)) and p1 not in startingPoints:
                startingPoints.append(p1)
            if (self.isLeaf(p2) or self.isTriple(p2)) and p2 not in startingPoints:
                startingPoints.append(p2)
        return startingPoints
    
    def pathStartingFrom(self, place, path):
        maxLength = len(path)
        longestPath = path.copy()
        length = 0
        for edge in self.ownedStreets:
            p1, p2 = edge
            if(place in edge and (edge not in path)):
                newPath = path.copy()
                if(p1 == place):
                    newPath.append(edge)
                    newPath, length = self.pathStartingFrom(p2, newPath)
                elif(p2 == place):
                    newPath.append(edge)
                    newPath, length = self.pathStartingFrom(p1, newPath)
                if(length > maxLength):
                    maxLength = length
                    longestPath = newPath.copy()
        return longestPath, len(longestPath)

    def longestStreet(self):
        leaves = self.findStartingPoints()
        # print("Riga 342 player, player id ", self.id, "number of owned streets: ", len(self.ownedStreets))
        maxLength = 0
        for leaf in leaves:
            longestPath, length = self.pathStartingFrom(leaf, [])
            if(length > maxLength):
                maxLength = length
        # print("Riga 355 player ", self.id, "longestPath: ", longestPath, "Lunghezza: ", maxLength)
        return maxLength


