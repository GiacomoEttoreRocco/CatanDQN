import Classes as c
import pygame
import Graphics.GameView as GameView
import time
import os
import csv


speed = True
withDelay = False
realPlayer = False
save = True

def goNextIfInvio(speed = False):
    if(not speed):
        event = pygame.event.wait()
        while event.type != pygame.KEYDOWN:
            event = pygame.event.wait()
    else:
        event = pygame.event.get()
        pygame.display.update()
    view.updateGameScreen()
    if(withDelay):
        time.sleep(0.2)

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
                saveBoardAndGlobals(save, f1, f2, player) ######################################################
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
                    saveBoardAndGlobals(save, f1, f2, player) ######################################################
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
            saveBoardAndGlobals(save, f1, f2, player) ######################################################
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
            saveBoardAndGlobals(save, f1, f2, player) ######################################################
    else:
        game.dice_production(dicesValue)
    move, thingNeeded = game.bestMove(player, turnCardUsed)
    move(player, thingNeeded)
    saveBoardAndGlobals(save, f1, f2, player) ######################################################
    goNextIfInvio(speed)
    print("Player ", player.id, " mossa: ", move, " ")
    if(game.checkWon(player)):
        return
    if(move in c.Move.cardMoves()):
            turnCardUsed = True
    while(move != c.Move.passTurn and not game.checkWon(player)): # move è una funzione 
        move, thingNeeded = game.bestMove(player, turnCardUsed)
        move(player, thingNeeded)

        saveBoardAndGlobals(save, f1, f2, player) ######################################################

        goNextIfInvio(speed)
        print("Player ", player.id, " mossa: ", move, " ")
        if(move in c.Move.cardMoves()):
            turnCardUsed = True

def playGameWithGraphic(game, view):
    GameView.GameView.setupAndDisplayBoard(view)
    GameView.GameView.setupPlaces(view)
    GameView.GameView.updateGameScreen(view)
    pygame.display.update()
    turn = 0 
    won = False
    # START INIZIALE
    if(realPlayer == True):
        game.players[3].AI = False
    for p in game.players:
        game.doInitialChoise(p)
        saveBoardAndGlobals(save, f1, f2, p) #################

        goNextIfInvio(speed)
    for p in sorted(game.players, reverse=True):
        game.doInitialChoise(p, giveResources = True)
        saveBoardAndGlobals(save, f1, f2, p) #################

    # INITIAL CHOISE TERMINATED

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


def openCsvGlobal():
    sourceFileDir = os.path.dirname(os.path.abspath(__file__))
    csvPath = os.path.join(sourceFileDir, "globalFeatures.csv")
    gFeatureCsv = open(csvPath, "w")
    return gFeatureCsv

def openCsvBoard():
    sourceFileDir = os.path.dirname(os.path.abspath(__file__))
    csvPath = os.path.join(sourceFileDir, "graphPreEmbdding.csv")
    f = open(csvPath, "w")
    return f

def saveBoard(f):
    c.Board.Board().stringForCsv(f)

def gFeaturePlayerToCsv(gfc, player):
        writer = csv.writer(gfc)
        gFeatures = str(player.id)+","+str(player.victoryPoints)+","+str(player.usedKnights)+","+str(player.resources["crop"])+"," \
            +str(player.resources["iron"])+","+str(player.resources["wood"])+","+str(player.resources["clay"])+","+str(player.resources["sheep"])
        writer.writerow([gFeatures])

def saveBoardAndGlobals(save, fglobal, fboard, player):
    if(save):
        gFeaturePlayerToCsv(fglobal, player)
        saveBoard(fboard)

def closeFiles(f1, f2):
    f1.close()
    f2.close()


f1 = openCsvGlobal()
f2 = openCsvBoard()


g = c.Game.Game()
view = GameView.GameView(g)
playGameWithGraphic(g, view)

closeFiles(f1, f2)


