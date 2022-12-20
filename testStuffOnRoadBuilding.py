import Classes.Player as Player
import Classes.Board as Board
import Classes.Bank as Bank
import Classes.Move as Move
import Classes.Game as Game
import unittest

class TestRoadBuilding(unittest.TestCase):
    def __init__(self, *args, **kwargs):
            super(TestRoadBuilding, self).__init__(*args, **kwargs)
            self.initializeStuff()

    def initializeStuff(self):
        self.game = Game.Game(2)
        self.Player1 = self.game.players[0]
        self.Player2 = self.game.players[1]

    def test1(self):
        previous =  list.copy(self.Player1.ownedStreets)
        Move.useRoadBuildingCard(self.Player1, [(0, 1), (0, 8)])
        self.assertEqual(len(previous) + 2, len(self.Player1.ownedStreets))
        self.assertTrue((0, 1) in self.Player1.ownedStreets)
        self.assertTrue((0, 8) in self.Player1.ownedStreets)

    def test2(self):
        Move.placeFreeStreet(self.Player1,(7,8))
        Move.placeFreeStreet(self.Player1,(0,8))
        Move.placeFreeStreet(self.Player1,(0,1))
        Move.placeFreeStreet(self.Player1,(1,2))
        Move.placeFreeStreet(self.Player1,(2,3))

        print("Vp 1: ", self.Player1.victoryPoints)
        print("Vp 2: ", self.Player2.victoryPoints)

        Move.placeFreeStreet(self.Player2,(4,5))
        Move.placeFreeStreet(self.Player2,(5,6))
        Move.placeFreeStreet(self.Player2,(6,14))
        Move.placeFreeStreet(self.Player2,(14,15))

        print("Player2: ", self.Player2.ownedStreets)

        Move.useRoadBuildingCard(self.Player2, ((15,25), (25,26)) )

        print("Player2: ", self.Player2.ownedStreets)

        print("Vp 1: ", self.Player1.victoryPoints)
        print("Vp 2: ", self.Player2.victoryPoints)

if __name__ == '__main__':
    unittest.main()