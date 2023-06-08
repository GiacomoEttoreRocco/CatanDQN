from AI.Gnn import Gnn
from Classes import Bank, Board
from Classes.Strategy.StrategySL import StrategySL
from Command import commands, controller

class HybridStrategy(StrategySL):
    def name(self):
        return "SL-HYBRID"
    
    def __init__(self):
        pass
    
    def bestAction(self, player):
        if(player.game.actualTurn<player.game.nplayers):
            actions = [commands.FirstChoiseCommand]
        elif(player.game.actualTurn<player.game.nplayers*2):
            actions = [commands.SecondChoiseCommand]
        else:
            actions = self.availableActions(player.turnCardUsed)
        max = -1
        thingsNeeded = None
        bestAction = actions[0]
        for action in actions: 
            evaluation, tempInput = self.evaluate(action, player)
            if(max <= evaluation):
                max = evaluation
                thingsNeeded = tempInput
                bestAction = action
        onlyPassTurn = commands.PassTurnCommand in actions and len(actions)==1
        return bestAction, thingsNeeded, onlyPassTurn
    
    def evaluate(self, action, player):
        return super().evaluate(action, player)
    
    def actionValue(self, player, action, thingNeeded = None):
        ctr = controller.ActionController()
        if(action == commands.FirstChoiseCommand or action == commands.SecondChoiseCommand or action == commands.PlaceInitialStreetCommand or action == commands.PlaceInitialColonyCommand or action == commands.PlaceSecondColonyCommand):
            ctr.execute(action(player, thingNeeded)) 
            toRet = Gnn.Gnn().evaluatePositionForPlayer(player)
            ctr.undo()
        else:
            if(action == commands.PassTurnCommand): 
                toRet = Gnn.Gnn().evaluatePositionForPlayer(player)
            else:
                pointsBefore = player.victoryPoints
                previousPossibleColonies = player.calculatePossibleColony()
                previousCount = player.resourceCount()
                ctr.execute(action(player, thingNeeded)) 
                if(player.victoryPoints >= 10):
                    ctr.undo()
                    return 1000.0
                if(pointsBefore < player.victoryPoints):
                    # print("Greedy path, player: ", self.id)
                    toRet = 300.0 + Gnn.Gnn().evaluatePositionForPlayer(player)
                elif(action == commands.PlaceStreetCommand or action == commands.PlaceFreeStreetCommand):
                    if(previousPossibleColonies == []):
                        if(player.calculatePossibleColony() != []):
                            val = Gnn.Gnn().evaluatePositionForPlayer(player)
                            toRet = 190.0 + val
                        else:
                            val = Gnn.Gnn().evaluatePositionForPlayer(player)
                            toRet = 185.0 + val
                    elif(len(previousPossibleColonies) < len(self.calculatePossibleColony())):
                            val = Gnn.Gnn().evaluatePositionForPlayer(player)
                            toRet = 170.0 + val
                    else:
                        toRet = Gnn.Gnn().evaluatePositionForPlayer(player)
                elif(action == commands.TradeBankCommand):
                    if(player.resources['crop'] > 1 and player.resources['iron'] > 2 and player.calculatePossibleCity() != []):
                        toRet = 175.0 + Gnn.Gnn().evaluatePositionForPlayer(player)
                    elif(player.resources['crop'] > 0 and player.resources['iron'] > 0 and player.resources['sheep'] > 0):
                        toRet = 175.0 + Gnn.Gnn().evaluatePositionForPlayer(player)
                    elif(player.resources['wood'] > 0 and player.resources['clay'] > 0 and player.resources['crop'] > 0 and player.resources['sheep'] > 0 and previousPossibleColonies != []):
                        toRet = 175.0 + Gnn.Gnn().evaluatePositionForPlayer(player)
                    elif(player.resources['wood'] > 0 and player.resources['clay'] > 0 and previousCount >= 7):
                        toRet = 175.0 + Gnn.Gnn().evaluatePositionForPlayer(player)
                    elif(previousCount >= 8):
                        toRet = 100.0 + Gnn.Gnn().evaluatePositionForPlayer(player)
                    else:
                        toRet = Gnn.Gnn().evaluatePositionForPlayer(player)
                else:
                    toRet = Gnn.Gnn().evaluatePositionForPlayer(player)
                ctr.undo()
        return toRet 
