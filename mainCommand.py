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

speed = False
# speed = True
withDelay = False
realPlayer = False
save = True
ctr = controller.ActionController()

WINNERS = [0.0, 0.0, 0.0, 0.0]
devCardsBought = [0.0, 0.0, 0.0, 0.0]

# def goNext():
#     if(not speed):
#         event = pygame.event.wait()
#         while event.type != pygame.KEYDOWN:
#             event = pygame.event.wait()
#     else:
#         event = pygame.event.get()
#     pygame.display.update()
#     view.updateGameScreen()

def decisionManager(player, action):
    if(not speed):
        event = pygame.event.wait()
        while event.type != pygame.KEYDOWN:
            event = pygame.event.wait()
        if(event.key == pygame.K_a):
            print("The player", str(player.id), "is going to do an AI move.")
            player.AI = True
            player.RANDOM = False
            ctr.execute(action)
        elif(event.key == pygame.K_r):
            print("The player", str(player.id), "is going to do a random move.")
            player.AI = False
            player.RANDOM = True
            ctr.execute(action)
        elif(event.key == pygame.K_u):
            ctr.undo()
        elif(event.key == pygame.K_k):
            ctr.redo()
        else:
            print(event.key)
    else:
        ctr.execute(action)
        event = pygame.event.get()

    view.updateGameScreen()
    pygame.display.update()

def doInitialChoice(player: c.PlayerWithCommands, giveResources = False):
    if(player.AI or player.RANDOM):
        evaluation, colonyChoosen = player.evaluate(commands.PlaceInitialColonyCommand)
        # ctr.execute(commands.PlaceInitialColonyCommand(player, colonyChoosen))
        decisionManager(player, commands.PlaceInitialColonyCommand(player, colonyChoosen))
        if(giveResources):
            for touchedResource in c.Board.Board().places[colonyChoosen.id].touchedResourses:
                c.Bank.Bank().giveResource(player, touchedResource)
        # print("Initial choise, colony: ", str(colonyChoosen.id))
        evaluation, edgeChoosen = player.evaluate(commands.PlaceInitialStreetCommand)
        # ctr.execute(commands.PlaceInitialStreetCommand(player, edgeChoosen))
        decisionManager(player, commands.PlaceInitialStreetCommand(player, edgeChoosen))

    # else:
    #     actions = []
    #     for colony in player.calculatePossibleInitialColony():
    #         actions.append((commands.PlaceInitialColonyCommand, colony))
    #     action, colonyChoosen = player.chooseAction(actions)
    #     self.ctr.execute(action(player, colonyChoosen))
    #     if(giveResources):
    #         for touchedResource in Board.Board().places[colonyChoosen.id].touchedResourses:
    #             Bank.Bank().giveResource(player, touchedResource)
    #     actions = []
    #     for street in player.calculatePossibleInitialStreets():
    #         actions.append((commands.PlaceInitialStreetCommand, street))
    #     action, edgeChoosen = player.chooseAction(actions)
    #     self.ctr.execute(action(player, edgeChoosen))

def doActionDecisions(game: c.GameWithCommands, player: c.PlayerWithCommands):
    if(game.checkWon(player)):
        return
    view.updateGameScreen()
    action, thingNeeded, lengthActions = game.bestAction(player, player.turnCardUsed)
    if action == commands.PlaceStreetCommand  or action == commands.PlaceColonyCommand:
        previousLongestStreetOwner = player.game.longestStreetPlayer(False)
        decisionManager(player, action(player, thingNeeded))
        checkLongestStreetOwner(previousLongestStreetOwner, player)  
    elif action == commands.UseKnightCommand:
        decisionManager(player, action(player, thingNeeded))
        decisionManager(player, commands.StealResourceCommand(player, c.Board.Board().tiles[thingNeeded]))  
    else:
        # ctr.execute(action(player, thingNeeded))
        decisionManager(player, action(player, thingNeeded))

    if(action == commands.BuyDevCardCommand):
        devCardsBought[player.id-1] += 1
    if(lengthActions > 1):
        saveMove(save, player) 
    if(game.checkWon(player)):
        return
    if(action in commands.cardCommands()):
        player.turnCardUsed = True

def playGameWithGraphic(game: c.GameWithCommands, view=None):
    oldPlayerTurnId = 0
    global devCardsBought
    devCardsBought = [0.0, 0.0, 0.0, 0.0]
    GameView.GameView.setupAndDisplayBoard(view)
    GameView.GameView.setupPlaces(view)
    GameView.GameView.updateGameScreen(view)
    pygame.display.update()
    game.actualTurn = 0 
    won = False
    # START INIZIALE
    #if(speed):
    game.players[0].AI = True
    # game.players[1].AI = True
    # game.players[2].AI = True
    # game.players[3].AI = True
    # game.players[0].RANDOM = True
    game.players[1].RANDOM = True
    game.players[2].RANDOM = True
    game.players[3].RANDOM = True
    
    for p in game.players:
        doInitialChoice(p)
        GameView.GameView.updateGameScreen(view)
        saveMove(save, p) #################
        #goNext()
    for p in sorted(game.players, reverse=True):
        doInitialChoice(p, giveResources = True)
        GameView.GameView.updateGameScreen(view)
        saveMove(save, p) #################
        #goNext()

    while won == False:
        playerTurn = game.players[game.actualTurn%game.nplayer]
        if(oldPlayerTurnId != playerTurn.id):
            decisionManager(playerTurn, commands.InitialTurnSetupCommand(playerTurn))
            dicesValue = playerTurn.game.dices[playerTurn.game.actualTurn]
            if(dicesValue == 7):
                playerTurn.game.sevenOnDices()
                ev, pos = playerTurn.evaluate(commands.UseRobberCommand)
                decisionManager(playerTurn, commands.UseRobberCommand(playerTurn, pos))
                #goNext()
                decisionManager(playerTurn, commands.StealResourceCommand(playerTurn, c.Board.Board().tiles[pos]))
            else:
                decisionManager(playerTurn, commands.DiceProductionCommand(dicesValue, game))
        game.currentTurnPlayer = playerTurn
        doActionDecisions(game, playerTurn)
        if(playerTurn.victoryPoints >= 10):
            WINNERS[playerTurn.id-1] += 1
            s = 'Winner: ' + str(playerTurn.id) + "\n"
            game.printVictoryPointsOfAll(devCardsBought)
            saveToCsv(playerTurn)
            print(s) 
            won = True
        oldPlayerTurnId = playerTurn.id
    #goNext()
    pygame.quit()

def saveMove(save, player):
    if(save):
        places = c.Board.Board().placesToDict(player)
        edges = c.Board.Board().edgesToDict(player)
        globals = player.globalFeaturesToDict()
        total.loc[len(total)] = [places, edges, globals]

def saveToCsv(victoryPlayer):
    for i in range(len(total)):
        total.globals[i]['winner'] = victoryPlayer.id
        #print("Winner saved: ", str(victoryPlayer.id))
    print(total.globals[0])
    print(total.globals[1])
    print(total.globals[2])
    print(total.globals[3])

    print("Length of total moves of this game: ", len(total))
    global allGames
    allGames = pd.concat([allGames, total], ignore_index=True)
    print("Length of total moves of allGames: ", len(allGames))

def checkLongestStreetOwner(previousLongestStreetOwner: c.PlayerWithCommands, player: c.PlayerWithCommands):
    actualLongestStreetOwner = player.game.longestStreetPlayer(False)
    if(previousLongestStreetOwner != actualLongestStreetOwner):
        player.game.longestStreetOwner = actualLongestStreetOwner
        actualLongestStreetOwner.victoryPoints += 2
        #print("-2 riga 21")
        previousLongestStreetOwner.victoryPoints -= 2

###########################################################################################################################

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

###########################################################################################################################

epochs = 1
batchs = 10

train = True

for epoch in range(epochs):
    print('Iteration: ', epoch+1, "/", epochs)
    allGames = pd.DataFrame(data={'places': [], 'edges':[], 'globals':[]})   
    for batch in range(batchs): 
        print('game: ', batch+1, "/", batchs) 
        total = pd.DataFrame(data={'places': [], 'edges':[], 'globals':[]})
        #g = c.Game.Game()
        g = c.GameWithCommands.Game()
        view = GameView.GameView(g)
        playGameWithGraphic(g, view)
        c.Board.Board().reset()
        c.Bank.Bank().reset()

    if(train):
        allGames.to_json("./json/game.json")
        Gnn.Gnn().trainModel()
        printWinners()