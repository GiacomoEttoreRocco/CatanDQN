import Classes.Player as Player
import Classes.Board as Board
import Classes.Bank as Bank
import Classes.Move as Move
import Classes.Game as Game
import unittest

class TestVictoryPoints(unittest.TestCase):
    def __init__(self, *args, **kwargs):
            super(TestVictoryPoints, self).__init__(*args, **kwargs)
            self.initializeStuff()

    def initializeStuff(self):
        self.game = Game.Game(2)
        
        self.Player1 = self.game.players[0]
        self.Player2 = self.game.players[1]

    def testUndoColony(self):
        Move.placeInitialColony(self.Player1, Board.Board().places[0])
        Move.placeInitialColony(self.Player1, Board.Board().places[0], True)
        self.assertEqual(self.Player1.victoryPoints, 0)

    def testUndoCity(self):
        Bank.Bank().giveResource(self.Player1, "crop")
        Bank.Bank().giveResource(self.Player1, "crop")
        Bank.Bank().giveResource(self.Player1, "iron")
        Bank.Bank().giveResource(self.Player1, "iron")
        Bank.Bank().giveResource(self.Player1, "iron")
        Move.placeCity(self.Player1, Board.Board().places[0])
        Bank.Bank().giveResource(self.Player1, "crop")
        Bank.Bank().giveResource(self.Player1, "crop")
        Bank.Bank().giveResource(self.Player1, "iron")
        Bank.Bank().giveResource(self.Player1, "iron")
        Bank.Bank().giveResource(self.Player1, "iron")
        Move.placeCity(self.Player1, Board.Board().places[0], True)
        self.assertEqual(self.Player1.victoryPoints, 0)

    def testColony(self):
        Move.placeInitialColony(self.Player1, Board.Board().places[0])
        self.assertEqual(self.Player1.victoryPoints, 1)

    def testCity(self):
        Bank.Bank().giveResource(self.Player1, "crop")
        Bank.Bank().giveResource(self.Player1, "clay")
        Bank.Bank().giveResource(self.Player1, "wood")
        Bank.Bank().giveResource(self.Player1, "sheep")

        Move.placeColony(self.Player1, Board.Board().places[0])
        Move.placeInitialColony(self.Player1, Board.Board().places[3])
        
        Bank.Bank().giveResource(self.Player1, "crop")
        Bank.Bank().giveResource(self.Player1, "crop")
        Bank.Bank().giveResource(self.Player1, "iron")
        Bank.Bank().giveResource(self.Player1, "iron")
        Bank.Bank().giveResource(self.Player1, "iron")

        Move.placeCity(self.Player1, Board.Board().places[0])
        self.assertEqual(self.Player1.victoryPoints, 3)

    

if __name__ == '__main__':
    unittest.main()