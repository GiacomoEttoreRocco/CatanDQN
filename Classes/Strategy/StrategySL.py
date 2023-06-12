from Classes import Bank, Board
from Classes.Strategy.Strategy import Strategy
from Command import commands


class StrategySL(Strategy):
    def name():
        pass

    def bestAction(self, player):
        ...
        pass

    def actionValue(self, player, action, thingNeeded):
        print("StrategySL")
        ...
        pass

    def chooseParameters(self, action, player):
            if(action == commands.DiscardResourceCommand):
                possibleCards = [r for r in player.resources.keys() if player.resources[r] > 0]
                candidateCard = None
                max = -1
                for card in possibleCards:
                    valutation = self.actionValue(player, action, card)
                    if(max < valutation):
                        max = valutation
                        candidateCard = card
                return max, candidateCard
        
            if(action == commands.PlaceStreetCommand or action == commands.PlaceFreeStreetCommand):
                possibleEdges = player.calculatePossibleStreets()
                candidateEdge = None
                max = -1
                for edge in possibleEdges: 
                    valutation = self.actionValue(player, action, edge)
                    if(max < valutation):
                        max = valutation
                        candidateEdge = edge
                return max, candidateEdge

            if(action == commands.PlaceInitialStreetCommand):
                possibleEdges = player.calculatePossibleInitialStreets()
                candidateEdge = None
                max = -1
                for edge in possibleEdges: 
                    valutation = self.actionValue(player, action, edge)
                    if(max < valutation):
                        max = valutation
                        candidateEdge = edge
                return max, candidateEdge

            if(action == commands.PlaceInitialColonyCommand or action == commands.FirstChoiseCommand or action == commands.SecondChoiseCommand or action == commands.PlaceSecondColonyCommand):
                possibleColony = player.calculatePossibleInitialColony()
                candidateColony = None
                max = -1
                for colony in possibleColony:
                    valutation = self.actionValue(player, action, colony)
                    if(max < valutation):
                        max = valutation
                        candidateColony = colony
                return max, candidateColony    

            if(action == commands.PlaceColonyCommand):
                possibleColony = player.calculatePossibleColonies()
                candidateColony = None
                max = -1
                for colony in possibleColony:
                    valutation = self.actionValue(player, action, colony)
                    if(max < valutation):
                        max = valutation
                        candidateColony = colony
                return max, candidateColony

            if(action == commands.PlaceCityCommand):
                possibleCity = player.calculatePossibleCities()
                candidateCity = None
                max = -1
                for city in possibleCity:
                    valutation = self.actionValue(player, action, city)
                    if(max < valutation):
                        max = valutation
                        candidateCity = city
                return max, candidateCity            

            if(action == commands.BuyDevCardCommand):
                valutation = self.actionValue(player, action, None)
                return valutation, None

            if(action == commands.PassTurnCommand):
                return self.actionValue(player, action), None

            if(action == commands.UseKnightCommand):
                max = -1
                for tile in Board.Board().tiles: 
                    if(tile.identificator != Board.Board().robberTile):
                        valutation = self.actionValue(player, action, tile.identificator)
                        if(max < valutation):
                            max = valutation
                            candidatePos = tile.identificator
                return max, candidatePos

            if(action == commands.UseRobberCommand): # Yes they are the same method, but must be differentiated becouse of the count of knights.
                max = -1
                for tile in Board.Board().tiles: 
                    if(tile.identificator != Board.Board().robberTile):
                        valutation = self.actionValue(player, action, tile.identificator)
                        if(max < valutation):
                            max = valutation
                            candidatePos = tile.identificator
                return max, candidatePos        

            if(action == commands.TradeBankCommand):
                possibleTrades = player.calculatePossibleTrades()
                candidateTrade = None
                max = -1
                for trade in possibleTrades:
                    valutation = self.actionValue(player, action, trade)
                    if(max < valutation):
                        max = valutation
                        candidateTrade = trade
                return max, candidateTrade

            if(action == commands.UseMonopolyCardCommand):
                max = -1
                for res in Bank.Bank().resources.keys():
                    valutation = self.actionValue(player, action, res)
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
                            valutation = self.actionValue(player, action, (res1, res2))
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
                if len(player.ownedStreets) < 14:
                    possibleEdges = player.calculatePossibleStreets()
                    max1 = -1
                    for edge in possibleEdges: 
                        valutation = self.actionValue(player, commands.PlaceFreeStreetCommand, edge)
                        if(max1 < valutation):
                            max1 = valutation
                            candidateEdge1 = edge
                    toRet += max1
                if len(player.ownedStreets) < 15:
                    possibleEdges = player.calculatePossibleStreets()
                    max2 = -1
                    for edge in possibleEdges: 
                        if(edge != candidateEdge1):
                            valutation = self.actionValue(player, commands.PlaceFreeStreetCommand, edge)
                            if(max2 < valutation):
                                max2 = valutation
                                candidateEdge2 = edge
                    toRet += max2
                return toRet, [candidateEdge1, candidateEdge2]