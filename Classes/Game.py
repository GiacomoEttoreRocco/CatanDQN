import Player
from Board import Board
from Bank import Bank
import Move
import random
import time

class Game:
    def __init__(self, num_players = 4):

        self.nplayer = num_players
        self.players = [Player.Player(i, self) for i in range(1, num_players+1)]
        self.largestArmyPlayer = 0

    def dice_production(self, number):
        for tile in Board().tiles:
            if tile.number == number:
                for p in tile.associatedPlaces:
                    if(Board().places[p].owner != 0):
                        if(Board().places[p].isColony):
                            print(self.players[Board().places[p].owner-1].id, "has taken a ", tile.resource, "\n")
                            Bank().giveResource(self.players[Board().places[p].owner-1], tile.resource)
                        elif(Board().places[p].isCity):
                            print(self.players[Board().places[p].owner-1].id, "has taken 2 ", tile.resource, "\n")
                            Bank().giveResource(self.players[Board().places[p].owner-1], tile.resource)
                            Bank().giveResource(self.players[Board().places[p].owner-1], tile.resource)

    def bestMove(self, player: Player, usedCard):
        moves = player.availableMoves(usedCard)
        max = 0
        thingsNeeded = None
        bestMove = Move.passTurn
        for move in moves:
            #print("Debug riga 34 in game best move just calculate (not done): ", move)
            evaluation, tempInput = player.evaluate(move)
            if(max <= evaluation):
                max = evaluation
                thingsNeeded = tempInput
                bestMove = move
        return bestMove, thingsNeeded

    def sevenOnDices(self, player: Player):
        for pyr in self.players:
            if(pyr.totalCards() >= 7):
                nCards = int(pyr.totalCards()/2)
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
        moves = player.availableMoves(turnCardUsed)
        if(Move.useKnight in moves):
            actualEvaluation = Board().actualEvaluation()
            afterKnight, place = player.evaluate(Move.useKnight)
            if(afterKnight > actualEvaluation):
                Move.useKnight(player, place)
                print("BEFORE roll dice: ", Move.useKnight, "\n")
                turnCardUsed = True 

        dicesValue = self.rollDice()
        print("Dice value: ", dicesValue)
        for tile in Board().tiles:
            if(tile.number == dicesValue):
                print(tile.resource)


        if(dicesValue == 7):
            print("############# SEVEN! #############")
            self.sevenOnDices(player)
        else:
            self.dice_production(dicesValue)

        move, thingNeeded = self.bestMove(player, turnCardUsed)

        #print("Player ", player.id, " debug riga 89 GAME, prima mossa: ", move, "\n")

        while(move != Move.passTurn): # move è una funzione 

            #player.printStats()
            move(player, thingNeeded)
            print("Player ", player.id, " mossa: ", move, " ")
            player.printStats()


            move, thingNeeded = self.bestMove(player, turnCardUsed)
        
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
        max = 0
        belonger = 0
        if(self.largestArmyPlayer != 0):
            max = self.players[self.largestArmyPlayer].usedKnights
            belonger = self.largestArmyPlayer
        for p in self.players:
            if(p.usedKnights >= 3 and p.usedKnights > max):
                max = p.usedKnights 
                belonger = p
        if(not justCheck):
            self.largestArmyPlayer = belonger
        return belonger
            
    def playGame(self):
        turn = 1 
        won = False
        # START INIZIALE
        for p in self.players:
            self.doInitialChoise(p)
            #print("Debug riga 118 Game, End first initial choise of player: ", p.id)

        for p in sorted(self.players, reverse=True):
            self.doInitialChoise(p)
            #print("Debug riga 122 Game, End second initial choise of player: ", p.id)
            p.printStats()
    

        #print("---------------------------------------- debug: riga 124 player, fine initial choises. ----------------------------------------")
        # START INIZIALE FINITO

        while won == False:
            playerTurn = self.players[turn%self.nplayer]
            if(playerTurn.victoryPoints >= 10):
                won = True
                print("Il vincitore è il Player ", str(playerTurn.id), "!")
            self.doTurn(playerTurn)
            #print("----------------------------------- End turn player: ", playerTurn.id, "-------------------------------------------------------------------")
            time.sleep(5)
            turn += 1
            if(turn % 4 == 0):
                print("========================================== Start of turn: ", str(int(turn/4)), "=========================================================")

g = Game()
g.playGame()








                        
        