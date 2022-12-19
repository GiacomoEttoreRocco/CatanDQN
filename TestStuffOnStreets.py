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
        
        
        self.results = [5, 5, 11, 5, 10, 10, 5, 7, 6, 10, 5, 10, 5]

        if(test == 0): #green color
            Move.placeFreeStreet(self.Player1, (41,49))
            Move.placeFreeStreet(self.Player1, (39,40))
            Move.placeFreeStreet(self.Player1, (40,41))
            Move.placeFreeStreet(self.Player1, (41,42))
            Move.placeFreeStreet(self.Player1, (42,43))
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
            Move.placeFreeStreet(self.Player1, (39, 47))

        if(test == 3):
            Move.placeFreeStreet(self.Player1, (28,38))
            Move.placeFreeStreet(self.Player1, (38,39))
            Move.placeFreeStreet(self.Player1, (39,40))
            Move.placeFreeStreet(self.Player1, (40,41))
            Move.placeFreeStreet(self.Player1, (41,42))

            Move.placeInitialColony(self.Player2, Board.Board().places[42])

            Move.placeFreeStreet(self.Player1, (42,43))
            Move.placeFreeStreet(self.Player1, (43,44))

        if(test == 4):
            Move.placeFreeStreet(self.Player1, (39,40))
            Move.placeFreeStreet(self.Player1, (40,41))

            Move.placeFreeStreet(self.Player1, (41, 49))

            Move.placeFreeStreet(self.Player2, (41,42)) # Player2

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
            Move.placeFreeStreet(self.Player2, (41, 49)) # Player2
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

            Move.placeInitialColony(self.Player1, Board.Board().places[42])

            Move.placeFreeStreet(self.Player1, (42,43))
            Move.placeFreeStreet(self.Player1, (43,44))        

        if(test == 7):
            Move.placeFreeStreet(self.Player1, (39,40))
            Move.placeFreeStreet(self.Player1, (40,41))
            Move.placeFreeStreet(self.Player1, (41,49))
            Move.placeFreeStreet(self.Player1, (48,49))
            Move.placeFreeStreet(self.Player1, (47,48))   
            Move.placeFreeStreet(self.Player1, (39,47)) 
            Move.placeFreeStreet(self.Player1, (30,40)) 
        
        if(test == 8):
            Move.placeFreeStreet(self.Player1, (39,40))
            Move.placeFreeStreet(self.Player1, (40,41))
            Move.placeFreeStreet(self.Player1, (41,49))
            Move.placeFreeStreet(self.Player1, (48,49))
            Move.placeFreeStreet(self.Player1, (47,48))  
            Move.placeFreeStreet(self.Player1, (39,47))

        if(test == 9):
            Move.placeFreeStreet(self.Player1, (39,40))
            Move.placeFreeStreet(self.Player1, (40,41))
            Move.placeFreeStreet(self.Player1, (41,49))
            Move.placeFreeStreet(self.Player1, (48,49))
            Move.placeFreeStreet(self.Player1, (47,48))  
            Move.placeFreeStreet(self.Player1, (39,47))

            Move.placeFreeStreet(self.Player1, (22,23))
            Move.placeFreeStreet(self.Player1, (23,24))
            Move.placeFreeStreet(self.Player1, (24,35))
            Move.placeFreeStreet(self.Player1, (34,35))
            Move.placeFreeStreet(self.Player1, (33,34))  
            Move.placeFreeStreet(self.Player1, (22,33))

            Move.placeFreeStreet(self.Player1, (19,20))
            Move.placeFreeStreet(self.Player1, (20,21))
            Move.placeFreeStreet(self.Player1, (11,21))
            
            Move.placeFreeStreet(self.Player1, (9,19))
            Move.placeFreeStreet(self.Player1, (21,22))
               
        if(test == 10):
            Move.placeFreeStreet(self.Player1, (4,5))
            Move.placeFreeStreet(self.Player1, (5,6))
            Move.placeFreeStreet(self.Player1, (6,14))
            Move.placeFreeStreet(self.Player1, (15,25))
            Move.placeFreeStreet(self.Player1, (25,26))
            Move.placeFreeStreet(self.Player1, (26,37))
            Move.placeFreeStreet(self.Player1, (36,37))
            Move.placeFreeStreet(self.Player1, (36,46))

        if(test == 11):
            print('Step 0 --------- Owner: ', self.game.longestStreetOwner.id, ' length: ', self.game.longestStreetLength)
            for p in self.game.players:
                print('player: ', p.id, ' points: ', p.victoryPoints)

            Move.placeFreeStreet(self.Player2, (4,5))
            Move.placeFreeStreet(self.Player2, (5,6))
            Move.placeFreeStreet(self.Player2, (6,14))
            Move.placeFreeStreet(self.Player2, (15,25))
            Move.placeFreeStreet(self.Player2, (25,26))
            Move.placeFreeStreet(self.Player2, (26,37))
            Move.placeFreeStreet(self.Player2, (36,37))
            Move.placeFreeStreet(self.Player2, (36,46))

            print('Step 1 --------- Owner: ', self.game.longestStreetOwner.id, ' length: ', self.game.longestStreetLength)
            for p in self.game.players:
                print('player: ', p.id, ' points: ', p.victoryPoints)

            Move.placeFreeStreet(self.Player1, (39,40))
            Move.placeFreeStreet(self.Player1, (40,41))
            Move.placeFreeStreet(self.Player1, (41,49))
            Move.placeFreeStreet(self.Player1, (48,49))
            Move.placeFreeStreet(self.Player1, (47,48))  
            Move.placeFreeStreet(self.Player1, (39,47))

            Move.placeFreeStreet(self.Player1, (22,23))
            Move.placeFreeStreet(self.Player1, (23,24))
            Move.placeFreeStreet(self.Player1, (24,35))
            Move.placeFreeStreet(self.Player1, (34,35))
            Move.placeFreeStreet(self.Player1, (33,34))  
            Move.placeFreeStreet(self.Player1, (22,33))

            Move.placeFreeStreet(self.Player1, (19,20))
            Move.placeFreeStreet(self.Player1, (20,21))
            Move.placeFreeStreet(self.Player1, (11,21))
            
            Move.placeFreeStreet(self.Player1, (9,19))
            Move.placeFreeStreet(self.Player1, (21,22))

            print('Step 2 --------- Owner: ', self.game.longestStreetOwner.id, ' length: ', self.game.longestStreetLength)
            for p in self.game.players:
                print('player: ', p.id, ' points: ', p.victoryPoints)

        if(test == 12):
            print('Step 0 --------- Owner: ', self.game.longestStreetOwner.id, ' length: ', self.game.longestStreetLength)
            for p in self.game.players:
                print('player: ', p.id, ' points: ', p.victoryPoints)

            Move.placeFreeStreet(self.Player2, (4,5))
            Move.placeFreeStreet(self.Player2, (5,6))
            Move.placeFreeStreet(self.Player2, (6,14))
            Move.placeFreeStreet(self.Player2, (15,25))
            Move.placeFreeStreet(self.Player2, (25,26))
            Move.placeFreeStreet(self.Player2, (26,37))
            Move.placeFreeStreet(self.Player2, (36,37))
            Move.placeFreeStreet(self.Player2, (36,46))

            print('Step 1 --------- Owner: ', self.game.longestStreetOwner.id, ' length: ', self.game.longestStreetLength)
            for p in self.game.players:
                print('player: ', p.id, ' points: ', p.victoryPoints)

            Move.placeFreeStreet(self.Player1, (39,40))
            Move.placeFreeStreet(self.Player1, (40,41))
            Move.placeFreeStreet(self.Player1, (41,49))
            Move.placeFreeStreet(self.Player1, (48,49))
            Move.placeFreeStreet(self.Player1, (47,48))  
            Move.placeFreeStreet(self.Player1, (39,47))

            Move.placeFreeStreet(self.Player1, (22,23))
            Move.placeFreeStreet(self.Player1, (23,24))
            Move.placeFreeStreet(self.Player1, (24,35))
            Move.placeFreeStreet(self.Player1, (34,35))
            Move.placeFreeStreet(self.Player1, (33,34))  
            Move.placeFreeStreet(self.Player1, (22,33))

            Move.placeFreeStreet(self.Player1, (19,20))
            Move.placeFreeStreet(self.Player1, (20,21))
            Move.placeFreeStreet(self.Player1, (11,21))
            
            Move.placeFreeStreet(self.Player1, (9,19))
            Move.placeFreeStreet(self.Player1, (21,22))

            print('Step 2 --------- Owner: ', self.game.longestStreetOwner.id, ' length: ', self.game.longestStreetLength)
            for p in self.game.players:
                print('player: ', p.id, ' points: ', p.victoryPoints)

            Move.placeInitialColony(self.Player2, Board.Board().places[21])
            Move.placeInitialColony(self.Player2, Board.Board().places[33])
            Move.placeInitialColony(self.Player2, Board.Board().places[23])
            Move.placeInitialColony(self.Player2, Board.Board().places[40])
            Move.placeInitialColony(self.Player2, Board.Board().places[48])

            Move.placeFreeStreet(self.Player2, (3,4))

            print('Step 3 --------- Owner: ', self.game.longestStreetOwner.id, ' length: ', self.game.longestStreetLength)
            for p in self.game.players:
                print('player: ', p.id, ' points: ', p.victoryPoints)
            


            
            


        
        # assert self.game.longestStreetLength == results[test], f'Was: {str(self.game.longestStreetLength)} Aspected: {str(results[test])}'

for n in [0,1,2,3,4,5,6,7,8,9,10,11,12]:
    Board.Board().reset()
    ts = TestStreetOwner(n)
    # print(ts.game.connectedPlacesToPlace(ts.Player1, 24))
    # print(Board.Board().edges[(23,24)])
    assert ts.game.longestStreetLength == ts.results[n] , f'Was {ts.game.longestStreetLength} espected: {ts.results[n]}'
    

# print(ts.game.findLeaves(ts.Player1))


