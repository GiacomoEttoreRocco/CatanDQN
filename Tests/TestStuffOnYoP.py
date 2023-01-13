import Classes.Player as Player
import Classes.Board as Board
import Classes.Bank as Bank
import Classes.Move as Move
import Classes.Game as Game
import os

class TestYearOfPlenty():    

    def __init__(self, test):
        print("Evaluating test: ", test)
        self.game = Game.Game(2)
        self.largestArmyPlayer = Player.Player(0, self.game)
        
        self.Player1 = self.game.players[0]
        self.Player2 = self.game.players[1]
        
        
        self.results = [18, 17, 0, 1, 0]

        #Check if player has 2 resources and if the bank gives them. Check Bank.resources after the move
        #To buy: buyDevCard(player, card, undo = False):

        def emptyBank(self, resource):
            print("Resource amount before bank emptied: ", Bank.Bank().resources)
            for i in range(19):
                Bank.Bank().giveResource(self.Player2, resource)  #Player2 just stacks resources for test purposes
            print("Resource amount after bank emptied: ", Bank.Bank().resources)

        if(test == 0):  #Generally tests if it works for two different resources
            print("--Before request--")
            print("Resource on", str(self.Player1.id), "for first resource :" , self.Player1.resources["crop"])
            print("Resource on", str(self.Player1.id), "for second resource :" , self.Player1.resources["iron"])
            print("Resource left on bank for first resource :" , Bank.Bank().resources["crop"])
            print("Resource left on bank for first resource :" , Bank.Bank().resources["iron"])

            Move.useYearOfPlentyCard(self.Player1, ["crop", "iron"])

            print("--After request--")
            print("Resource on", str(self.Player1.id), "for first resource :" , self.Player1.resources["crop"])
            print("Resource on", str(self.Player1.id), "for second resource :" , self.Player1.resources["iron"])
            print("Resource left on bank for first resource :" , Bank.Bank().resources["crop"])
            print("Resource left on bank for second resource :" , Bank.Bank().resources["iron"])

        if(test == 1):  #Generally tests if it works for same resource
            print("--Before request--")
            print("Resource on", str(self.Player1.id), "for resource :" , self.Player1.resources["crop"])
            print("Resource left on bank for resource :" , Bank.Bank().resources["crop"])
            Move.useYearOfPlentyCard(self.Player1, ["crop", "crop"])
            print("--After request--")
            print("Resource on", str(self.Player1.id), "for resource :" , self.Player1.resources["crop"])
            print("Resource left on bank for first resource :" , Bank.Bank().resources["crop"])

        if(test == 2):   #Tests that card doesn't work when Bank is out of resources for a given resource
            emptyBank(self, "crop")
            print("--Before request--")
            print("Resource on player ", str(self.Player1.id), "is :" , self.Player1.resources["crop"])
            Move.useYearOfPlentyCard(self.Player1, ["crop", "crop"])
            print("--After request--")
            print("Resource on player ", str(self.Player1.id), "is :" , self.Player1.resources["crop"])

        if(test == 3):   #Tests that card doesn't work when Bank is out of resources for two different resources
            emptyBank(self, "crop")
            print("--Before request--")
            print("First resource on player ", str(self.Player1.id), "is: " , self.Player1.resources["crop"])
            print("Second resource on player ", str(self.Player1.id), "is: " , self.Player1.resources["iron"])
            Move.useYearOfPlentyCard(self.Player1, ["crop", "iron"])
            print("--After request--")
            print("First resource on player ", str(self.Player1.id), "is: " , self.Player1.resources["crop"])
            print("Second resource on player ", str(self.Player1.id), "is: " , self.Player1.resources["iron"])

        if(test == 4):  #Tests if amount of yop cards is decreased after usage
            print("Amount of yop cards before purchase: ", self.Player1.yearOfPlentyCard)
            Move.buyDevCard(self.Player1, "year_of_plenty")
            self.Player1.yearOfPlentyCard += 1  #Since doTurn is not being called, we simulate the increase
            print("Amount of yop cards after purchase: ", self.Player1.yearOfPlentyCard)
            Move.useYearOfPlentyCard(self.Player1, ["crop", "crop"])
            print("Amount of yop cards after usage: ", self.Player1.yearOfPlentyCard)
    
n = 0

TestYearOfPlenty(n)