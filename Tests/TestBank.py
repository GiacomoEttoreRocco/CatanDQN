import unittest
import Classes as c
from Classes.Bank import Bank

class TestBank(unittest.TestCase):

    def testGiveResource(self):
        previousResources = Bank().resources['wood']
        Bank().giveResource(c.Player.Player(1, None), 'wood')
        actualResources = Bank().resources['wood']
        self.assertEqual(previousResources-1, actualResources)

    def testResourceToAsk(self):
        #res = "wood"
        player = c.Player.Player(1, None)

        player.ownedHarbors = []
        val = Bank().resourceToAsk(player, "wood")
        self.assertEqual(val, 4)

        player.ownedHarbors.append("2:1 wood")
        val = Bank().resourceToAsk(player, "wood")
        self.assertEqual(val, 2)

        player.ownedHarbors = []
        player.ownedHarbors.append("3:1")
        val = Bank().resourceToAsk(player, "wood")
        self.assertEqual(val, 3)

if __name__ == '__main__':
    unittest.main()

    