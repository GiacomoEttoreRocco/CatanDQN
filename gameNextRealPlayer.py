import Classes as c
import pygame
import Graphics.GameView as GameView
import time

speed = True
realPlayer = False
def goNextIfInvio(speed = False):
    if(not speed):
        event = pygame.event.wait()
        while event.type != pygame.KEYDOWN:
            event = pygame.event.wait()
    else:
        event = pygame.event.get()
        pygame.display.update()
    view.updateGameScreen()
    time.sleep(1)

def doTurnGraphic(game: c.Game, player: c.Player):
    turnCardUsed = False 
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
        if(player.AI == True):
            actualEvaluation = c.Board.Board().actualEvaluation()
            afterKnight, place = player.evaluate(c.Move.useKnight)
            if(afterKnight > actualEvaluation):
                c.Move.useKnight(player, place)
                print("BEFORE ROLL DICE: ", c.Move.useKnight, "\n")
                view.updateGameScreen()
                turnCardUsed = True 
        else:
            view.updateGameScreen() 
            if(player.unusedKnights >= 1 and not turnCardUsed):
                print("Mosse disponibili: ")
                moves = [(c.Move.passTurn, None)]
                for i in range(0, 17):
                    moves.append((c.Move.useKnight, i))  
                for i, move in enumerate(moves):
                    print("Move ", i, ": ", move)  
                toDo = int(input("Inserisci l'indice della mossa che vuoi eseguire: "))
                if(toDo != 0):
                    c.Move.useKnight(player, moves[toDo][1])
                else:
                    c.Move.passTurn(player)
                
    if(game.checkWon(player)):
        return
    ####################################################################### ROLL DICES #####################################################################   
    dicesValue = game.rollDice()
    game.dice = dicesValue
    ########################################################################################################################################################
    print("Dice value: ", dicesValue)
    if(dicesValue == 7):
        game.sevenOnDices()
        print("############# SEVEN! #############")
        if(player.AI == True):
            ev, pos = player.evaluate(c.Move.useRobber)
            c.Move.useRobber(player, pos)
        else:
            print("Mosse disponibili: ")
            moves = []
            for i in range(0, 19):
                if(i != c.Board.Board().robberTile):
                    moves.append((c.Move.useRobber, i))  
            for i, move in enumerate(moves):
                print("Move ", i, ": ", move)  
            toDo = int(input("Inserisci l'indice della mossa che vuoi eseguire: "))
            c.Move.useRobber(player, moves[toDo][1])
    else:
        game.dice_production(dicesValue)
    move, thingNeeded = game.bestMove(player, turnCardUsed)
    move(player, thingNeeded)
    goNextIfInvio(speed)
    print("Player ", player.id, " mossa: ", move, " ")
    if(game.checkWon(player)):
        return
    if(move in c.Move.cardMoves()):
            turnCardUsed = True
    while(move != c.Move.passTurn and not game.checkWon(player)): # move è una funzione 
        move, thingNeeded = game.bestMove(player, turnCardUsed)
        move(player, thingNeeded)
        goNextIfInvio(speed)
        print("Player ", player.id, " mossa: ", move, " ")
        if(move in c.Move.cardMoves()):
            turnCardUsed = True

def playGameWithGraphic(game, view):
    GameView.GameView.setupAndDisplayBoard(view)
    GameView.GameView.setupPlaces(view)
    GameView.GameView.updateGameScreen(view)
    pygame.display.update()
    turn = 1 
    won = False
    # START INIZIALE
    if(realPlayer == True):
        game.players[3].AI = False

    for p in game.players:
        game.doInitialChoise(p)
        goNextIfInvio(speed)
    for p in sorted(game.players, reverse=True):
        game.doInitialChoise(p, giveResources = True)
        goNextIfInvio(speed)
    while won == False:
        playerTurn = game.players[turn%game.nplayer]
        game.currentTurn = playerTurn
        turn += 1
        doTurnGraphic(game, playerTurn)
        if(playerTurn.victoryPoints >= 10):
            return playerTurn
        if(turn % 4 == 0):
            print("========================================== Start of turn: ", str(int(turn/4)), "=========================================================")
    goNextIfInvio(speed)
    pygame.quit()

g = c.Game.Game()
view = GameView.GameView(g)
playGameWithGraphic(g, view)

