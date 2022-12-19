import Classes.Player as Player
import Classes.Board as Board
import Classes.Bank as Bank
import Classes.Move as Move
import Classes.Game as Game
import os

class TestStreetOwner():    

    def __init__(self, test):
        print("Evaluating test: ", test)
        self.game = Game.Game(2)
        self.longestStreetOwner = Player.Player(0, self.game)
        
        self.Player1 = self.game.players[0]
        self.Player2 = self.game.players[1]
        
        
        self.results = [5,5,11,5,9,11,5, 7]

        if(test == 0): #green color
            Move.placeFreeStreet(self.Player1, (41,49))
            Move.placeFreeStreet(self.Player1, (39,40))
            Move.placeFreeStreet(self.Player1, (40,41))
            Move.placeFreeStreet(self.Player1, (41,42))
            Move.placeFreeStreet(self.Player1, (42,43))
            
            os.system('cls' if os.name == 'nt' else 'clear')
            Move.placeFreeStreet(self.Player1, (43,44))
            # Move.placeColony(self.Player2, Board.Board().places[41])


        if(test == 1): #green color
            Move.placeFreeStreet(self.Player1, (39,40))
            Move.placeFreeStreet(self.Player1, (40,41))
            Move.placeFreeStreet(self.Player1, (41,42))
            Move.placeFreeStreet(self.Player1, (42,43))
            Move.placeFreeStreet(self.Player1, (43,44))
        if(test == 2):
            Move.placeFreeStreet(self.Player1, (39,40))
            Move.placeFreeStreet(self.Player1, (40,41))

            Move.placeFreeStreet(self.Player1, (41, 49)) # quello in mezzo

            Move.placeFreeStreet(self.Player1, (41,42))
            Move.placeFreeStreet(self.Player1, (42,43))
            Move.placeFreeStreet(self.Player1, (43,51))

            Move.placeFreeStreet(self.Player1, (50, 51))
            Move.placeFreeStreet(self.Player1, (49, 50))
            Move.placeFreeStreet(self.Player1, (48, 49))
            Move.placeFreeStreet(self.Player1, (47, 48))
            os.system('cls' if os.name == 'nt' else 'clear')
            Move.placeFreeStreet(self.Player1, (39, 47))

        if(test == 3):
            Move.placeFreeStreet(self.Player1, (28,38))
            Move.placeFreeStreet(self.Player1, (38,39))
            Move.placeFreeStreet(self.Player1, (39,40))
            Move.placeFreeStreet(self.Player1, (40,41))
            Move.placeFreeStreet(self.Player1, (41,42))

            Move.placeColony(self.Player2, Board.Board().places[42])

            Move.placeFreeStreet(self.Player1, (42,43))
            os.system('cls' if os.name == 'nt' else 'clear')
            Move.placeFreeStreet(self.Player1, (43,44))

        if(test == 4):
            Move.placeFreeStreet(self.Player1, (39,40))
            Move.placeFreeStreet(self.Player1, (40,41))

            Move.placeFreeStreet(self.Player1, (41, 49)) # quello in mezzo

            Move.placeFreeStreet(self.Player2, (41,42)) 

            Move.placeFreeStreet(self.Player1, (42,43))
            Move.placeFreeStreet(self.Player1, (43,51))

            Move.placeFreeStreet(self.Player1, (50, 51))
            Move.placeFreeStreet(self.Player1, (49, 50))
            Move.placeFreeStreet(self.Player1, (48, 49))
            Move.placeFreeStreet(self.Player1, (47, 48))
            Move.placeFreeStreet(self.Player1, (39, 47))

        if(test == 5):
            Move.placeFreeStreet(self.Player1, (39,40))
            Move.placeFreeStreet(self.Player1, (40,41))

            Move.placeFreeStreet(self.Player2, (41, 49)) # quello in mezzo

            Move.placeFreeStreet(self.Player1, (41,42)) 

            Move.placeFreeStreet(self.Player1, (42,43))
            Move.placeFreeStreet(self.Player1, (43,51))

            Move.placeFreeStreet(self.Player1, (50, 51))
            Move.placeFreeStreet(self.Player1, (49, 50))
            Move.placeFreeStreet(self.Player1, (48, 49))
            Move.placeFreeStreet(self.Player1, (47, 48))
            Move.placeFreeStreet(self.Player1, (39, 47))

        if(test == 6):
            Move.placeFreeStreet(self.Player1, (39,40))
            Move.placeFreeStreet(self.Player1, (40,41))
            Move.placeFreeStreet(self.Player1, (41,42))

            Move.placeColony(self.Player1, Board.Board().places[42])

            Move.placeFreeStreet(self.Player1, (42,43))
            Move.placeFreeStreet(self.Player1, (43,44))        

        if(test == 7):
            Move.placeFreeStreet(self.Player1, (39,40))
            Move.placeFreeStreet(self.Player1, (40,41))
            Move.placeFreeStreet(self.Player1, (41,49))
            Move.placeFreeStreet(self.Player1, (48,49))
            Move.placeFreeStreet(self.Player1, (47,48))   
            Move.placeFreeStreet(self.Player1, (39,47)) 
            os.system('cls' if os.name == 'nt' else 'clear')
            Move.placeFreeStreet(self.Player1, (30,40)) 

        # assert self.game.longestStreetLength == results[test], f'Was: {str(self.game.longestStreetLength)} Aspected: {str(results[test])}'

for n in [0,1,2,7]:
    ts = TestStreetOwner(n)
    print(ts.game.findLeaves(ts.Player1))
    assert ts.game.longestStreetLength == ts.results[n] , f'Was {ts.game.longestStreetLength} espected: {ts.results[n]}'
    

# print(ts.game.findLeaves(ts.Player1))


