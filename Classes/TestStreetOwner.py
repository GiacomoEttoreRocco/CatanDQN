# import Player
# from Bank import Bank
# from Board import Board
# import Move
# import Game

# class TestStreetOwner():    

#     def __init__(self):
#         self.game = Game.Game(1)
#         self.longestStreetOwner = Player.Player(0, self.game)
#         Player1 = self.game.players[0]
#         Move.placeFreeStreet(Player1, (39,40))
#         Move.placeFreeStreet(Player1, (40,41))
#         Move.placeFreeStreet(Player1, (41,42))
#         Move.placeFreeStreet(Player1, (42,43))
#         Move.placeFreeStreet(Player1, (43,44))

#         self.visitedEdges = []

#     def longestStreetPlace(self, player, newPlace, oldPlace):

#         placesToVisit = list.copy(Board().graph.listOfAdj[newPlace])
#         placesToVisit.remove(oldPlace)
#         max = 0
#         for p in placesToVisit:
#             edge = tuple(sorted([p, newPlace]))
#             print("Inside : ", edge)
#             if(Board().edges[edge] == player.id and edge not in self.visitedEdges):
#                 self.visitedEdges.append(edge)
#                 print("Visited edges: ", self.visitedEdges)
#                 actual = 1 + self.longestStreetPlace(player, p, newPlace)
#                 if(max < actual):
#                     max = actual
#         return max
              
#     def longestStreet(self, player, edge):
#         self.visitedEdges = [edge]
#         print("Edge: ", edge)
#         #print("1 " + str(self.longestStreetPlace(player, edge[0], edge[1])) + " " + str(self.longestStreetPlace(player, edge[1], edge[0])))
#         toRet =  1 + self.longestStreetPlace(player, edge[0], edge[1]) + self.longestStreetPlace(player, edge[1], edge[0])
#         print("Tourette: ", toRet)
#         return toRet

#     def calculateLongestStreet(self, player):
#         max = 0
#         for edge in player.ownedStreets:
#             actual = self.longestStreet(player, edge)
#             print(actual)
#             if(max < actual):
#                 max = actual
#         return max

#     def longestStreetPlayer(self, justCheck = False):
#         max = 4
#         belonger = self.longestStreetOwner
#         for p in self.game.players:
#             actual = self.calculateLongestStreet(p)
#             if(max < actual):
#                 max = actual
#                 belonger = p
#         if(not justCheck):
#             if(belonger != self.longestStreetOwner):
#                 self.longestStreetOwner = belonger
#         print("SDGNSJGBIGBG BELONGER ", self.longestStreetOwner.id, "  New one: " , belonger.id)
#         return belonger

# ts = TestStreetOwner()
# ts.longestStreetPlayer(False)