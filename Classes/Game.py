import Player
from Board import Board
from Bank import Bank
import Move
import random
import time

class Game:
    def __init__(self, num_players = 4):

        self.nplayer = num_players
        self.players = [Player.Player(i+1, self) for i in range(0, num_players)]
        self.largestArmyPlayer = Player.Player(0, self)

    def dice_production(self, number):
        for tile in Board().tiles:
            if tile.number == number and tile != Board().robberTile:
                for p in tile.associatedPlaces:
                    if(Board().places[p].owner != 0):
                        if(Board().places[p].isColony):
                            print(self.players[Board().places[p].owner-1].id, "TAKEN ", tile.resource, "\n")
                            Bank().giveResource(self.players[Board().places[p].owner-1], tile.resource)
                        elif(Board().places[p].isCity):
                            print(self.players[Board().places[p].owner-1].id, "TAKEN 2 ", tile.resource, "\n")
                            Bank().giveResource(self.players[Board().places[p].owner-1], tile.resource)
                            Bank().giveResource(self.players[Board().places[p].owner-1], tile.resource)

    def bestMove(self, player: Player, usedCard):
        moves = player.availableMoves(usedCard)
        print("\n")
        player.printResources()
        print("\n AVAILABLE MOVES: ", moves, "\n")
        max = 0
        thingsNeeded = None
        bestMove = Move.passTurn
        for move in moves:
            evaluation, tempInput = player.evaluate(move)
            if(max <= evaluation):
                max = evaluation
                thingsNeeded = tempInput
                bestMove = move
        return bestMove, thingsNeeded

    def sevenOnDices(self, player: Player):
        for pyr in self.players:
            if(pyr.totalCards() >= 7):
                nCards = pyr.totalCards()/2
                while(pyr.totalCards() <= nCards):
                    eval, card = player.evaluate(Move.discardResource)
                    Move.discardResource(player, card)
                
        eval, tilePos = player.evaluate(Move.useRobber)
        Move.useRobber(player, tilePos)   

    def rollDice(self): 
        return random.randint(1,6) + random.randint(1,6)    

    def doTurn(self, player: Player):
        player.printStats()
        turnCardUsed = False # SI PUò USARE UNA SOLA CARTA SVILUPPO A TURNO.
        player.unusedKnights = player.unusedKnights + player.justBoughtKnights
        player.justBoughtKnights = 0
        player.monopolyCard += player.justBoughtMonopolyCard
        player.justBoughtMonopolyCard = 0
        player.roadBuildingCard += player.justBoughtRoadBuildingCard
        player.justBoughtRoadBuildingCard = 0
        player.yearOfPlentyCard += player.justBoughtYearOfPlentyCard
        player.justBoughtYearOfPlentyCard = 0
        if(player.unusedKnights > 0 and not turnCardUsed):
            actualEvaluation = Board().actualEvaluation()
            afterKnight, place = player.evaluate(Move.useKnight)
            if(afterKnight > actualEvaluation):
                Move.useKnight(player, place)
                print("BEFORE ROLL DICE: ", Move.useKnight, "\n")
                turnCardUsed = True 
        dicesValue = self.rollDice()
        print("Dice value: ", dicesValue)
        if(dicesValue == 7):
            print("############# SEVEN! #############")
        else:
            self.dice_production(dicesValue)

        move, thingNeeded = self.bestMove(player, turnCardUsed)
        print("Player ", player.id, " mossa: ", move, " ")

        move(player, thingNeeded)
        if(move in Move.cardMoves()):
                turnCardUsed = True

        while(move != Move.passTurn and player.victoryPoints < 10): # move è una funzione 
            move, thingNeeded = self.bestMove(player, turnCardUsed)
            print("Player ", player.id, " mossa: ", move, " ")
            move(player, thingNeeded)
            if(move in Move.cardMoves()):
                turnCardUsed = True
            #player.printStats()
        
    def doInitialChoise(self, player: Player):
        evaluation, colonyChoosen = player.evaluate(Move.placeFreeColony)
        Move.placeFreeColony(player, colonyChoosen)
        print("Initial choise, colony: ", str(colonyChoosen.id))

        evaluation, edgeChoosen = player.evaluate(Move.placeFreeStreet)
        Move.placeFreeStreet(player, edgeChoosen)
        print(" edge: ", edgeChoosen[0], " - ", edgeChoosen[1], "\n")
        print("edge owner: ", str(Board().edges[edgeChoosen]), "\n")

    def totalKnightsUsed(self):
        totKnightUsed = 0
        for p in self.players:
            totKnightUsed = totKnightUsed + p.usedKnights
        return totKnightUsed
    
    def largestArmy(self, justCheck = False):
        max = self.largestArmyPlayer.usedKnights
        belonger = self.largestArmyPlayer
        for p in self.players:
            if(p.usedKnights >= 3 and p.usedKnights > max):
                max = p.usedKnights 
                belonger = p
        if(not justCheck and belonger.id != 0):
            self.largestArmyPlayer = self.players[belonger.id-1]
        return belonger
            
    def playGame(self):
        turn = 1 
        won = False
        # START INIZIALE
        for p in self.players:
            self.doInitialChoise(p)

        for p in sorted(self.players, reverse=True):
            self.doInitialChoise(p)
            p.printStats()

        while won == False:
            playerTurn = self.players[turn%self.nplayer]
            #time.sleep(1)
            turn += 1
            if(playerTurn.victoryPoints >= 10):
                won = True
                playerTurn.printStats()
                print("Il vincitore è il Player ", str(playerTurn.id), "!")
                return
            else:
                self.doTurn(playerTurn)

            if(playerTurn.victoryPoints >= 10):
                won = True
                playerTurn.printStats()
                print("Il vincitore è il Player ", str(playerTurn.id), "!")
                return

            if(turn % 4 == 0):
                print("========================================== Start of turn: ", str(int(turn/4)), "=========================================================")
g = Game()
g.playGame()








                        
        