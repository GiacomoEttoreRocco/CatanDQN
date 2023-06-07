from AI.Gnn import Gnn
from Classes import Bank, Board
from Classes.Strategy.StrategySL import StrategySL
from Command import commands, controller

class PureStrategy(StrategySL):
    def name(self):
        return "PURE"

    def __init__(self):
        pass
    
    def bestAction(self, player):
        if(player.game.actualTurn<player.game.nplayers):
            actions = [commands.FirstChoiseCommand]
        elif(player.game.actualTurn<player.game.nplayers*2):
            actions = [commands.SecondChoiseCommand]
        else:
            actions = player.availableActions(player.turnCardUsed)
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
        if(action == commands.PassTurnCommand): 
            toRet = Gnn.Gnn().evaluatePositionForPlayer(player)
        else:
            previousCount = player.resourceCount()
            ctr.execute(action(player, thingNeeded)) 
            if(player.victoryPoints >= 10):
                # print("PureAI Agent conclusive move! " + str(action))
                ctr.undo()
                return 1000.0
            elif(previousCount >= 8 and player.resourceCount() < 8):
                toRet = 100.0 + Gnn.Gnn().evaluatePositionForPlayer(player)
                ctr.undo()
            else:
                toRet = Gnn.Gnn().evaluatePositionForPlayer(player)
                ctr.undo()
        return toRet # 