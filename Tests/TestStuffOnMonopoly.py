import Classes.Player as Player
import Classes.Board as Board
import Classes.Bank as Bank
import Classes.Move as Move
import Classes.Game as Game
import unittest

class TestMonopoly(unittest.TestCase):    

    def __init__(self, *args, **kwargs):
            super(TestMonopoly, self).__init__(*args, **kwargs)
            self.initializeStuff()

    def initializeStuff(self):
        self.game = Game.Game(4)
        
        self.activePlayer = self.game.players[0]
        self.otherPlayers = [self.game.players[1], self.game.players[2], self.game.players[3]]    

        #Check if players actually have their resources stolen by active player


    def givePlayersResources(self):
        for op in self.otherPlayers:
            Bank.Bank().giveResource(op, "crop")
            Bank.Bank().giveResource(op, "iron")
            Bank.Bank().giveResource(op, "sheep")
            Bank.Bank().giveResource(op, "clay")
            Bank.Bank().giveResource(op, "wood")

    def test_1(self):     #Generic placement
        self.givePlayersResources()
        print("--Before usage--")
        print("Active player resources: ", self.activePlayer.resources)
        Move.useMonopolyCard(self.activePlayer, "crop")
        Move.useMonopolyCard(self.activePlayer, "iron")
        print("--After usage--")
        print("Active player resources: ", self.activePlayer.resources)
        for op in self.otherPlayers:
            print("Other player ", op.id, " resources: ", op.resources)
            self.assertEquals(op.resources["crop"], 0)
            self.assertEquals(op.resources["iron"], 0)
        self.assertEquals(self.activePlayer.resources["crop"], 3)
        self.assertEquals(self.activePlayer.resources["iron"], 3)

    def test_2(self):
        print("--Before usage--")
        print("Active player resources: ", self.activePlayer.resources)
        Move.useMonopolyCard(self.activePlayer, "crop")
        Move.useMonopolyCard(self.activePlayer, "iron")
        print("--After usage--")
        print("Active player resources: ", self.activePlayer.resources)
        for op in self.otherPlayers:
            print("Other player ", op.id, " resources: ", op.resources)
        self.assertEquals(self.activePlayer.resources["crop"], 0)
        self.assertEquals(self.activePlayer.resources["iron"], 0)

            

if __name__ == '__main__':
    unittest.main()