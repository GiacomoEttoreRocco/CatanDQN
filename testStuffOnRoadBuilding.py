import Classes.Player as Player
import Classes.Board as Board
import Classes.Bank as Bank
import Classes.Move as Move
import Classes.Game as Game
import os

class TestRoadBuilding():    

    def __init__(self, test):
        print("Evaluating test: ", test)
        self.game = Game.Game(2)
        
        self.Player1 = self.game.players[0]
        self.Player2 = self.game.players[1]
        
        
        self.results = []

        #Check if player can build streets when they're out of space
        #To buy: buyDevCard(player, card, undo = False):

        if (test == 0):     #Generic placement
            print("Owned streets before usage: ", self.Player1.ownedStreets)
            Move.useRoadBuildingCard(self.Player1, [(0, 1), (0, 8)])
            print("Owned streets after usage: ", self.Player1.ownedStreets)

        if (test == 1):
            edges = list(Board.Board().edges.keys())
            for i in range(14):
                Move.placeFreeStreet(self.Player1, edges[i])
            print(self.Player1.ownedStreets)
            Move.useRoadBuildingCard(self.Player1, [(10, 11), (11, 12)])
            print(self.Player1.ownedStreets)
            print(self.Player1.nStreets)




n = 1
TestRoadBuilding(n)