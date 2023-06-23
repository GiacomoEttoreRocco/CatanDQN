# import random
# from Classes import Bank, Board
# from Classes.Strategy.StrategySL import StrategySL
# from Command import commands, controller

# class PriorityStrategy(StrategySL):
#     def name(self):
#         return "PRIORITY"

#     def __init__(self):
#         pass

#     def bestAction(self, player):
#         if(player.game.actualTurn<player.game.nplayers):
#             actions = [commands.FirstChoiseCommand]
#         elif(player.game.actualTurn<player.game.nplayers*2):
#             actions = [commands.SecondChoiseCommand]
#         else:
#             actions = player.availableActions(player.turnCardUsed)
#         max = -1
#         thingsNeeded = None
#         bestAction = actions[0]
#         for action in actions: 
#             evaluation, tempInput = self.chooseParameters(action, player)
#             if(max <= evaluation):
#                 max = evaluation
#                 thingsNeeded = tempInput
#                 bestAction = action
#         onlyPassTurn = commands.PassTurnCommand in actions and len(actions)==1
#         return bestAction, thingsNeeded, onlyPassTurn

#     def chooseParameters(self, action, player):
#         return super().chooseParameters(action, player)
    
#     def actionValue(self, player, action, thingNeeded = None):
#         if(player._victoryPoints >= 8):
#             ctr = controller.ActionController()
#             ctr.execute(action(player, thingNeeded)) 
#             if(player._victoryPoints >= 10):
#                 toRet = 300.0
#                 ctr.undo()
#                 return toRet
#             ctr.undo()
#         if(action == commands.PassTurnCommand):
#             return 0.2 + random.uniform(0, 1)
#         if(action == commands.UseKnightCommand):
#             toRet = 1.5
#             return toRet + random.uniform(0,2)
#         if(action == commands.UseRobberCommand):
#             toRet = 1.5
#             return toRet + random.uniform(0,2)
#         if(action == commands.BuyDevCardCommand):
#             toRet = 1.5
#             return toRet + random.uniform(0,5)
#         if(action == commands.UseMonopolyCardCommand):
#             toRet = 100.0
#             return toRet
#         if(action == commands.PlaceStreetCommand or action == commands.PlaceInitialStreetCommand or action == commands.PlaceColonyCommand):
#             if(action == commands.PlaceColonyCommand):
#                 toRet = 90
#             else:
#                 toRet = 16
#             return toRet 
#         if(action == commands.PlaceInitialColonyCommand):
#             toRet = 10.0
#         elif(action == commands.PlaceStreetCommand): 
#             toRet = 10.0
#         elif(action == commands.PlaceCityCommand):
#             toRet = 100.0
#         elif(action == commands.TradeBankCommand):
#             toRet = 15.0
#         elif(action == commands.UseRoadBuildingCardCommand):
#             toRet = 2.0
#         elif(action == commands.UseYearOfPlentyCardCommand):
#             toRet = 200.0
#         elif(action == commands.DiscardResourceCommand):
#             toRet = 1.0 
#         else:
#             toRet = 0.5
#         return toRet + random.uniform(0,2)