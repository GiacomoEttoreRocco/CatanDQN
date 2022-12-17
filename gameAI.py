import Classes as c
import pygame
import Graphics.GameView as GameView
import os
import pandas as pd
import time
import AI.Gnn as Gnn

speed = True
withDelay = False
realPlayer = False # If you put this to true you will play with the 3th player.
save = True
total = pd.DataFrame(data={'places': [], 'edges':[], 'globals':[]})
allGames = pd.DataFrame(data={'places': [], 'edges':[], 'globals':[]})

def goNextIfInvio():
    if(not speed):
        event = pygame.event.wait()
        while event.type != pygame.KEYDOWN:
            event = pygame.event.wait()
    else:
        event = pygame.event.get()
        pygame.display.update()
    view.updateGameScreen()
    if(withDelay):
        time.sleep(0.05)

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
        if(player.AI or player.RANDOM):
            actualEvaluation = c.Board.Board().actualEvaluation()
            afterKnight, place = player.evaluate(c.Move.useKnight)
            if(afterKnight > actualEvaluation):
                c.Move.useKnight(player, place)
                saveMove(save, player) ######################################################
                # print("BEFORE ROLL DICE: ", c.Move.useKnight, "\n")
                view.updateGameScreen()
                turnCardUsed = True 
        else:
            view.updateGameScreen() 
            if(player.unusedKnights >= 1 and not turnCardUsed):
                # print("Mosse disponibili: ")
                moves = [(c.Move.passTurn, None)]
                for i in range(0, 17):
                    moves.append((c.Move.useKnight, i))  
                for i, move in enumerate(moves):
                    print("Move ", i, ": ", move)  
                toDo = int(input("Insert the move index of the move you want to play: "))
                if(toDo != 0):
                    c.Move.useKnight(player, moves[toDo][1])
                    saveMove(save, player) ######################################################
                else:
                    c.Move.passTurn(player)
                
    if(game.checkWon(player)):
        return
    ####################################################################### ROLL DICES #####################################################################   
    dicesValue = game.rollDice()
    game.dice = dicesValue
    ########################################################################################################################################################
    # print("Dice value: ", dicesValue)
    if(dicesValue == 7):
        game.sevenOnDices()
        # print("############# SEVEN! #############")
        if(player.AI or player.RANDOM):
            ev, pos = player.evaluate(c.Move.useRobber)
            c.Move.useRobber(player, pos)
            saveMove(save, player) ######################################################
        else:
            # print("Mosse disponibili: ")
            moves = []
            for i in range(0, 19):
                if(i != c.Board.Board().robberTile):
                    moves.append((c.Move.useRobber, i))  
            for i, move in enumerate(moves):
                print("Move ", i, ": ", move)  
            toDo = int(input("Inserisci l'indice della mossa che vuoi eseguire: "))
            c.Move.useRobber(player, moves[toDo][1])
            saveMove(save, player) ######################################################
    else:
        game.dice_production(dicesValue)
    move, thingNeeded = game.bestMove(player, turnCardUsed)
    move(player, thingNeeded)
    saveMove(save, player) ######################################################
    goNextIfInvio()
    # print("Player ", player.id, " mossa: ", move, " ")
    if(game.checkWon(player)):
        return
    if(move in c.Move.cardMoves()):
            turnCardUsed = True
    while(move != c.Move.passTurn and not game.checkWon(player)): # move è una funzione 
        move, thingNeeded = game.bestMove(player, turnCardUsed)
        move(player, thingNeeded)
        saveMove(save, player) ######################################################
        goNextIfInvio()
        # print("Player ", player.id, " mossa: ", move, " ")
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
    game.players[0].AI = True
    game.players[1].AI = True
    game.players[2].AI = True
    game.players[3].RANDOM = True
    for p in game.players:
        game.doInitialChoise(p)
        saveMove(save, p) #################

        goNextIfInvio()
    for p in sorted(game.players, reverse=True):
        game.doInitialChoise(p, giveResources = True)
        saveMove(save, p) #################
        # INITIAL CHOISE TERMINATED
        goNextIfInvio()
    while won == False:
        playerTurn = game.players[turn%game.nplayer]
        game.currentTurn = playerTurn
        turn += 1
        doTurnGraphic(game, playerTurn)
        if(playerTurn.victoryPoints >= 10):
            saveToCsv(playerTurn)
            return playerTurn
        # if(turn % 4 == 0):
            # print("========================================== Start of turn: ", str(int(turn/4)), "=========================================================")
    goNextIfInvio()
    pygame.quit()


def saveMove(save, player):
    if(save):
        places = c.Board.Board().placesToDict()
        edges = c.Board.Board().edgesToDict()
        globals = player.globalFeaturesToDict()

        total.loc[len(total)] = [places, edges, globals]

def saveToCsv(victoryPlayer):
    print('winner: ', victoryPlayer.id)
    for i in range(len(total)):
        total.globals[i]['winner'] = victoryPlayer.id
    print(len(total))
    global allGames
    allGames = pd.concat([allGames, total], ignore_index=True)

###########################################################################################################################
epochs = 100
batchs = 1

for epoch in range(epochs):
    print('epoch: ', epoch+1)
    allGames = pd.DataFrame(data={'places': [], 'edges':[], 'globals':[]})   
    for batch in range(batchs): 
        print('game: ', batch+1) 
        total = pd.DataFrame(data={'places': [], 'edges':[], 'globals':[]})
        g = c.Game.Game()
        view = GameView.GameView(g)
        playGameWithGraphic(g, view)
        c.Board.Board().reset()
        c.Bank.Bank().reset()
    allGames.to_json("./json/game.json")
    Gnn.Gnn().trainModel()