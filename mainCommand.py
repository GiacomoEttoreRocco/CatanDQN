import Classes as c
import Command.controller as controller
import Command.commands as commands
import Command.action as action
import pygame
import Graphics.GameView as GameView
import os
import pandas as pd
import time
import AI.Gnn as Gnn

speed = True
withGraphics = False
withDelay = False
realPlayer = False
save = True
train = True
ctr = controller.ActionController()

WINNERS = [0.0, 0.0, 0.0, 0.0]
devCardsBought = [0.0, 0.0, 0.0, 0.0]

def decisionManager(player):
    onlyPassTurn = True # if undo, we don't want them to be saved
    if(not speed):
        if(withGraphics):
            event = pygame.event.wait()
            while event.type != pygame.KEYDOWN:
                event = pygame.event.wait()
            if(event.key == pygame.K_a):
                player.AI = True
                player.RANDOM = False
                action, thingNeeded, onlyPassTurn = player.game.bestAction(player)
                ctr.execute(action(player, thingNeeded))
            elif(event.key == pygame.K_r):
                player.AI = False
                player.RANDOM = True
                action, thingNeeded, onlyPassTurn = player.game.bestAction(player)
                ctr.execute(action(player, thingNeeded))
            elif(event.key == pygame.K_u):
                ctr.undo()
            elif(event.key == pygame.K_k):
                ctr.redo()
            else:
                print(event.key)
    else:
        action, thingNeeded, onlyPassTurn = player.game.bestAction(player)
        ctr.execute(action(player, thingNeeded))
        if(withGraphics):
            event = pygame.event.get()
    if(withGraphics):
        view.updateGameScreen()
        pygame.display.update()
    if(not onlyPassTurn):  
        saveMove(save, player) 

def doActionDecisions(game: c.GameWithCommands, player: c.PlayerWithCommands, withGraphics = True):
    if(game.checkWon(player)):
        return
    if(withGraphics):
        view.updateGameScreen()

    previousLongestStreetOwner = player.game.longestStreetOwner
    action = decisionManager(player)
    if action == commands.PlaceStreetCommand  or action == commands.PlaceColonyCommand or action == commands.UseRoadBuildingCardCommand:  
        checkLongestStreetOwner(previousLongestStreetOwner, player) 

    if(action == commands.BuyDevCardCommand):
        devCardsBought[player.id-1] += 1

    if(game.checkWon(player)):
        return
    if(action in commands.cardCommands()):
        player.turnCardUsed = True

def playGameWithGraphic(game: c.GameWithCommands, view=None, withGraphics = True):
    # oldPlayerTurnId = 0
    global devCardsBought
    devCardsBought = [0.0, 0.0, 0.0, 0.0]
    if(withGraphics):
        GameView.GameView.setupAndDisplayBoard(view)
        GameView.GameView.setupPlaces(view)
        GameView.GameView.updateGameScreen(view)
        pygame.display.update()
    game.actualTurn = 0 
    won = False
    game.players[0].RANDOM = True
    game.players[1].RANDOM = True
    game.players[2].RANDOM = True
    game.players[3].RANDOM = True
    
    reverseTurnOffSet = {0 : 0, 1 : 1, 2 : 2, 3 : 3, 4 : 3, 5 : 2, 6 : 1, 7 : 0}

    while won == False:
        if(game.actualTurn < 8):
            playerTurn = game.players[reverseTurnOffSet[game.actualTurn]]     
            decisionManager(playerTurn)
            if(withGraphics):    
                GameView.GameView.updateGameScreen(view)
            saveMove(save, playerTurn) 
        else:
            playerTurn = game.players[game.actualTurn%game.nplayer]
            doActionDecisions(game, playerTurn, withGraphics)
            if(playerTurn.victoryPoints >= 10):
                WINNERS[playerTurn.id-1] += 1
                s = 'Winner: ' + str(playerTurn.id) + "\n"
                # game.printVictoryPointsOfAll(devCardsBought)
                saveToCsv(playerTurn)
                print(s) 
                won = True
    if(withGraphics):
        pygame.quit()

def saveMove(save, player):
    if(save):
        places = c.Board.Board().placesToDict(player)
        edges = c.Board.Board().edgesToDict(player)
        globals = player.globalFeaturesToDict()
        total.loc[len(total)] = [places, edges, globals]

def saveToCsv(victoryPlayer):
    if(save):
        for i in range(len(total)):
            total.globals[i]['winner'] = victoryPlayer.id
        print("Length of total moves of this game: ", len(total))
        global allGames
        allGames = pd.concat([allGames, total], ignore_index=True)
        print("Length of total moves of allGames: ", len(allGames))

def checkLongestStreetOwner(previousLongestStreetOwner: c.PlayerWithCommands, player: c.PlayerWithCommands):
    actualLongestStreetOwner = player.game.longestStreetPlayer()
    if(previousLongestStreetOwner != actualLongestStreetOwner):
        player.game.longestStreetOwner = actualLongestStreetOwner
        actualLongestStreetOwner.victoryPoints += 2
        previousLongestStreetOwner.victoryPoints -= 2

def printWinners():
    normValue = sum(WINNERS)
    toPrint = [0.0, 0.0, 0.0, 0.0]
    for i, v in enumerate(WINNERS):
        toPrint[i] = v/normValue
    s = ""
    for i, vperc in enumerate(toPrint):
        s = s + "Player " + str(i+1)+ ": " + str(round(vperc*100,2)) + " % "
    print(s)
    print(WINNERS)

epochs = 1
batchs = 1

for epoch in range(epochs):
    print('Iteration: ', epoch+1, "/", epochs)
    allGames = pd.DataFrame(data={'places': [], 'edges':[], 'globals':[]})   
    for batch in range(batchs): 
        print('game: ', batch+1, "/", batchs) 
        total = pd.DataFrame(data={'places': [], 'edges':[], 'globals':[]})
        #g = c.Game.Game()
        g = c.GameWithCommands.Game()
        if(withGraphics):
            view = GameView.GameView(g)
            playGameWithGraphic(g, view, withGraphics)
        else:
            playGameWithGraphic(g, None, withGraphics)
        c.Board.Board().reset()
        c.Bank.Bank().reset()

    if(train):
        allGames.to_json("./json/game.json")
        Gnn.Gnn().trainModel()
        printWinners()