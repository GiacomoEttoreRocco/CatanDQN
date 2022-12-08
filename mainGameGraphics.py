import Classes as c
import pygame
import Graphics.GameView as GameView

def doTurnGraphic(self, player: c.Player):
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

    view.updateGameScreen() 

    if(player.unusedKnights > 0 and not turnCardUsed):
        actualEvaluation = c.Board.Board().actualEvaluation()
        afterKnight, place = player.evaluate(c.Move.useKnight)
        if(afterKnight > actualEvaluation):
            c.Move.useKnight(player, place)
            print("BEFORE ROLL DICE: ", c.Move.useKnight, "\n")
            view.updateGameScreen()
            turnCardUsed = True 
    if(self.checkWon(player)):
        return
    ####################################################################### ROLL DICES #####################################################################   
    dicesValue = self.rollDice()
    ########################################################################################################################################################  
    print("Dice value: ", dicesValue)
    if(dicesValue == 7):
        print("############# SEVEN! #############")
    else:
        self.dice_production(dicesValue)

    move, thingNeeded = self.bestMove(player, turnCardUsed)
    move(player, thingNeeded)

    view.updateGameScreen()

    print("Player ", player.id, " mossa: ", move, " ")
    if(self.checkWon(player)):
        return

    if(move in c.Move.cardMoves()):
            turnCardUsed = True

    while(move != c.Move.passTurn and not self.checkWon(player)): # move è una funzione 

        move, thingNeeded = self.bestMove(player, turnCardUsed)
        move(player, thingNeeded)

        view.updateGameScreen()

        print("Player ", player.id, " mossa: ", move, " ")

        if(move in c.Move.cardMoves()):
            turnCardUsed = True

def playGameWithGraphic(self):

    view.displayGameScreen()  

    c.Board.Board().reset()
    c.Bank.Bank().reset()
    turn = 1 
    won = False
    # START INIZIALE
    for p in self.players:
        self.doInitialChoise(p)

        view.updateGameScreen()

    for p in sorted(self.players, reverse=True):
        self.doInitialChoise(p)

        view.updateGameScreen()

        p.printStats()
    while won == False:
        playerTurn = self.players[turn%self.nplayer]
        #time.sleep(5)
        turn += 1
        playerTurn.printStats()
        doTurnGraphic(self, playerTurn)
        if(playerTurn.victoryPoints >= 10):
            return playerTurn
        if(turn % 4 == 0):
            print("========================================== Start of turn: ", str(int(turn/4)), "=========================================================")


view = GameView.GameView()
g = c.Game.Game()
playGameWithGraphic(g)

