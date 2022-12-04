import unittest
import Player
import Move
from Bank import Bank

class TestMove(unittest.TestCase):
    def testTradeBank(self):
        player = Player.Player(1, None)
        player.resources["wood"] = 4

        previousWood = Bank().resources["wood"]
        previousCrop = Bank().resources["crop"]

        Move.tradeBank(player, ("crop", "wood"))

        self.assertEqual(player.resources["wood"], 0)
        self.assertEqual(player.resources["crop"], 1)

        self.assertEqual(previousWood+4, Bank().resources["wood"])
        self.assertEqual(previousCrop-1, Bank().resources["crop"])




if __name__ == '__main__':
    unittest.main()