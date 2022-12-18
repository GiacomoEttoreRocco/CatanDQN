import Classes.Player as Player
import Classes.Board as Board
import Classes.Bank as Bank
import Classes.Move as Move
import Classes.Game as Game

class TestStreetOwner():    

    def __init__(self, test):
        self.game = Game.Game(2)
        self.longestStreetOwner = Player.Player(0, self.game)
        
        Player1 = self.game.players[0]
        Player2 = self.game.players[1]

        result1 = 5
        result2 = 11
        result3 = 3
        result4 = 9
        result5 = 11
        result6 = 5


        if(test == 1): #green color
            Move.placeFreeStreet(Player1, (39,40))
            Move.placeFreeStreet(Player1, (40,41))
            Move.placeFreeStreet(Player1, (41,42))
            Move.placeFreeStreet(Player1, (42,43))
            Move.placeFreeStreet(Player1, (43,44))
        if(test == 2):
            Move.placeFreeStreet(Player1, (39,40))
            Move.placeFreeStreet(Player1, (40,41))

            Move.placeFreeStreet(Player1, (41, 49)) # quello in mezzo

            Move.placeFreeStreet(Player1, (41,42))
            Move.placeFreeStreet(Player1, (42,43))
            Move.placeFreeStreet(Player1, (43,51))

            Move.placeFreeStreet(Player1, (50, 51))
            Move.placeFreeStreet(Player1, (49, 50))
            Move.placeFreeStreet(Player1, (48, 49))
            Move.placeFreeStreet(Player1, (47, 48))
            Move.placeFreeStreet(Player1, (39, 47))
        if(test == 3):
            Move.placeFreeStreet(Player1, (28,38))
            Move.placeFreeStreet(Player1, (38,39))
            Move.placeFreeStreet(Player1, (39,40))
            Move.placeFreeStreet(Player1, (40,41))
            Move.placeFreeStreet(Player1, (41,42))

            Move.placeColony(Player2, Board.Board().places[42])

            Move.placeFreeStreet(Player1, (42,43))
            Move.placeFreeStreet(Player1, (43,44))

        if(test == 4):
            Move.placeFreeStreet(Player1, (39,40))
            Move.placeFreeStreet(Player1, (40,41))

            Move.placeFreeStreet(Player1, (41, 49)) # quello in mezzo

            Move.placeFreeStreet(Player2, (41,42)) 

            Move.placeFreeStreet(Player1, (42,43))
            Move.placeFreeStreet(Player1, (43,51))

            Move.placeFreeStreet(Player1, (50, 51))
            Move.placeFreeStreet(Player1, (49, 50))
            Move.placeFreeStreet(Player1, (48, 49))
            Move.placeFreeStreet(Player1, (47, 48))
            Move.placeFreeStreet(Player1, (39, 47))

        if(test == 5):
            Move.placeFreeStreet(Player1, (39,40))
            Move.placeFreeStreet(Player1, (40,41))

            Move.placeFreeStreet(Player2, (41, 49)) # quello in mezzo

            Move.placeFreeStreet(Player1, (41,42)) 

            Move.placeFreeStreet(Player1, (42,43))
            Move.placeFreeStreet(Player1, (43,51))

            Move.placeFreeStreet(Player1, (50, 51))
            Move.placeFreeStreet(Player1, (49, 50))
            Move.placeFreeStreet(Player1, (48, 49))
            Move.placeFreeStreet(Player1, (47, 48))
            Move.placeFreeStreet(Player1, (39, 47))

        if(test == 6):
            Move.placeFreeStreet(Player1, (39,40))
            Move.placeFreeStreet(Player1, (40,41))
            Move.placeFreeStreet(Player1, (41,42))

            Move.placeColony(Player1, Board.Board().places[42])

            Move.placeFreeStreet(Player1, (42,43))
            Move.placeFreeStreet(Player1, (43,44))         

        self.visitedEdges = []

    def longestStreetPlace(self, player, newPlace, oldPlace):

        placesToVisit = list.copy(Board().graph.listOfAdj[newPlace])
        placesToVisit.remove(oldPlace)
        max = 0
        for p in placesToVisit:
            edge = tuple(sorted([p, newPlace]))
            print("Inside : ", edge)
            if(Board().edges[edge] == player.id and edge not in self.visitedEdges):
                self.visitedEdges.append(edge)
                print("Visited edges: ", self.visitedEdges)
                actual = 1 + self.longestStreetPlace(player, p, newPlace)
                if(max < actual):
                    max = actual
        return max
              
    def longestStreet(self, player, edge):
        self.visitedEdges = [edge]
        print("Edge: ", edge)
        #print("1 " + str(self.longestStreetPlace(player, edge[0], edge[1])) + " " + str(self.longestStreetPlace(player, edge[1], edge[0])))
        toRet =  1 + self.longestStreetPlace(player, edge[0], edge[1]) + self.longestStreetPlace(player, edge[1], edge[0])
        print("Tourette: ", toRet)
        return toRet

    def calculateLongestStreet(self, player):
        max = 0
        for edge in player.ownedStreets:
            actual = self.longestStreet(player, edge)
            print(actual)
            if(max < actual):
                max = actual
        return max

    def longestStreetPlayer(self, justCheck = False):
        max = 4
        belonger = self.longestStreetOwner
        for p in self.game.players:
            actual = self.calculateLongestStreet(p)
            if(max < actual):
                max = actual
                belonger = p
        if(not justCheck):
            if(belonger != self.longestStreetOwner):
                self.longestStreetOwner = belonger
        print("SDGNSJGBIGBG BELONGER ", self.longestStreetOwner.id, "  New one: " , belonger.id)
        return belonger

ts = TestStreetOwner(3)
print(ts)