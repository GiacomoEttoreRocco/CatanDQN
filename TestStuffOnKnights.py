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
        self.largestArmyPlayer = Player.Player(0, self.game)
        
        self.Player1 = self.game.players[0]
        self.Player2 = self.game.players[1]
        self.Player3 = self.game.players[2]
        
        
        self.results = [0, 1]

        #Checks on game.largestArmyPlayer
        #useKnight(player, tilePosition, undo = False, justCheck = False):

        if(test == 0):
            Move.useKnight(self.Player1, 1)
            print(self.game.largestArmyPlayer.id)

        if(test == 1):
            Move.useKnight(self.Player1, 1)
            Move.useKnight(self.Player1, 2)
            Move.useKnight(self.Player1, 3)
            print(self.Player1.victoryPoints)
            print("Player: ", self.game.largestArmyPlayer.id)
            print(self.Player1.victoryPoints)


        

n = 1
TestLargestArmy(n)