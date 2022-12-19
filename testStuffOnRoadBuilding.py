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

    def test_1(self):
        previous =  list.copy(self.Player1.ownedStreets)
        Move.useRoadBuildingCard(self.Player1, [(0, 1), (0, 8)])
        self.assertEqual(len(previous) + 2, len(self.Player1.ownedStreets))
        self.assertTrue((0, 1) in self.Player1.ownedStreets)
        self.assertTrue((0, 8) in self.Player1.ownedStreets)

    # def test_2(self):
    #     edges = list(Board.Board().edges.keys())
    #     for i in range(14):
    #         Move.placeFreeStreet(self.Player1, edges[i])
    #     print(self.Player1.ownedStreets)
    #     Move.useRoadBuildingCard(self.Player1, [(10, 11), (11, 12)])
    #     print(self.Player1.ownedStreets)
    #     print(self.Player1.nStreets)





if __name__ == '__main__':
    unittest.main()