import Classes.Player as Player
import Classes.Board as Board
import Classes.Bank as Bank
import Classes.Move as Move
import Classes.Game as Game
import os

class TestLargestArmy():    

    def __init__(self, test):
        print("Evaluating test: ", test)
        self.game = Game.Game(3)
        b = Board.Board()

        self.largestArmyPlayer = Player.Player(0, self.game)
        
        self.Player1 = self.game.players[0]
        self.Player2 = self.game.players[1]
        self.Player3 = self.game.players[2]
        
        
        self.results = [0, 1]

        #Checks on game.largestArmyPlayer
        #useKnight(player, tilePosition, undo = False, justCheck = False):

        if(test == 0):
            Move.useKnight(self.Player1, 1)
            print("Largest army owner: ", self.game.largestArmyPlayer.id)


        if(test == 1):
            print("Player " + str(self.Player1.id) + " points before moves: ", self.Player1.victoryPoints)
            print("Largest army owner: ", self.game.largestArmyPlayer.id)

            Move.useKnight(self.Player1, 1)
            Move.useKnight(self.Player1, 2)
            Move.useKnight(self.Player1, 3)

            print("Player " + str(self.Player1.id) + " points after move: ", self.Player1.victoryPoints)
            print("Largest army owner: ", self.game.largestArmyPlayer.id)

        if(test == 2):
            for p in self.game.players:
                print("Player " + str(p.id) + " points before moves: ", p.victoryPoints)
            print("Largest army owner: ", self.game.largestArmyPlayer.id)

            Move.useKnight(self.Player1, 1)
            Move.useKnight(self.Player1, 2)
            Move.useKnight(self.Player1, 3)

            for p in self.game.players:
                print("Player " + str(p.id) + " points after first moves: ", p.victoryPoints)
            print("Largest army owner: ", self.game.largestArmyPlayer.id)


            Move.useKnight(self.Player2, 1)
            Move.useKnight(self.Player2, 2)
            Move.useKnight(self.Player2, 3)
            Move.useKnight(self.Player2, 4)

            for p in self.game.players:
                print("Player " + str(p.id) + " points after all moves: ", p.victoryPoints)
            print("Largest army owner: ", self.game.largestArmyPlayer.id)

        if(test == 3):
            for p in self.game.players:
                print("Player " + str(p.id) + " points before moves: ", p.victoryPoints)
            print("Largest army owner: ", self.game.largestArmyPlayer.id)

            Move.useKnight(self.Player1, 1, True)
            Move.useKnight(self.Player1, 2, True)
            Move.useKnight(self.Player1, 3, True)

            for p in self.game.players:
                print("Player " + str(p.id) + " points after first moves: ", p.victoryPoints)
            print("Largest army owner: ", self.game.largestArmyPlayer.id)

        if(test == 3):
            Move.placeInitialColony(self.Player1, Board.Board().places[0])
            Bank.Bank().giveResource(self.Player1, "clay")
            Move.useKnight(self.Player2, 0, True)
            print("Player1 resources: ", self.Player1.resources)
            print("Player2 resources: ", self.Player2.resources)



            


        

n = 3
TestLargestArmy(n)