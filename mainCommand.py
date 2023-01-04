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

# speed = False
speed = True

withDelay = False
realPlayer = False
save = True
ctr = controller.ActionController()

WINNERS = [0.0, 0.0, 0.0, 0.0]
devCardsBought = [0.0, 0.0, 0.0, 0.0]

def goNext():
    if(not speed):
        event = pygame.event.wait()
        while event.type != pygame.KEYDOWN:
            event = pygame.event.wait()
    else:
        event = pygame.event.get()
    pygame.display.update()
    view.updateGameScreen()


def doTurnGraphic(game: c.GameWithCommands, player: c.PlayerWithCommands):

    global devCardsBought
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
            actualEvaluation, thingNeeded = player.evaluate(commands.PassTurnCommand)
            afterKnight, tilePosition = player.evaluate(commands.UseKnightCommand)
            #print("Actual evaluation: ", actualEvaluation, " After knight: ", afterKnight)
            if(afterKnight > actualEvaluation):
                #print("pre dice roll knight")
                ctr.execute(commands.UseKnightCommand(player, tilePosition))
                goNext()
                ctr.execute(commands.StealResourceCommand(player, c.Board.Board().tiles[tilePosition]))
                saveMove(save, player) ######################################################
                view.updateGameScreen()
                turnCardUsed = True 
        # else:
        #     view.updateGameScreen() 
        #     if(player.unusedKnights >= 1 and not turnCardUsed):
        #         # print("Mosse disponibili: ")
        #         actions = [(commands.PassTurnCommand, None)]
        #         for i in range(0, 17):
        #             actions.append((commands.UseKnightCommand, i))  
        #         for i, move in enumerate(actions):
        #             print("Move ", i, ": ", move)  
        #         toDo = int(input("Insert the move index of the move you want to play: "))
        #         if(toDo != 0):
        #             ctr.execute(commands.UseKnightCommand(player, actions[toDo][1]))
        #             saveMove(save, player) ######################################################
        #         else:
        #             ctr.execute(commands.PassTurnCommand())
                
    if(game.checkWon(player)):
        return
    ####################################################################### ROLL DICES #####################################################################   
    dicesValue = game.dices[game.actualTurn]

    if(dicesValue == 7):
        game.sevenOnDices()
        if(player.AI or player.RANDOM):
            ev, pos = player.evaluate(commands.UseRobberCommand)
            ctr.execute(commands.UseRobberCommand(player, pos))
            goNext()
            ctr.execute(commands.StealResourceCommand(player, c.Board.Board().tiles[pos]))
            saveMove(save, player) 
        # else:
        #     actions = []
        #     for i in range(0, 19):
        #         if(i != c.Board.Board().robberTile):
        #             actions.append((commands.UseRobberCommand, i))  
        #     for i, action in enumerate(actions):
        #         print("Move ", i, ": ", action)  
        #     toDo = int(input("Inserisci l'indice della mossa che vuoi eseguire: "))
        #     ctr.execute(commands.UseRobberCommand(player, actions[toDo][1]))
        #     saveMove(save, player) 
    else:
        #game.dice_production(dicesValue)
        ctr.execute(commands.DiceProductionCommand(dicesValue, game))
        view.updateGameScreen()
    goNext()
    action, thingNeeded = game.bestAction(player, turnCardUsed)
    if action == commands.PlaceStreetCommand  or action == commands.PlaceColonyCommand:
        previousLongestStreetOwner = player.game.longestStreetPlayer(False)
        ctr.execute(action(player, thingNeeded, True))
        checkLongestStreetOwner(previousLongestStreetOwner, player)  
    elif action == commands.UseKnightCommand:
        ctr.execute(action(player, thingNeeded))
        goNext()
        ctr.execute(commands.StealResourceCommand(player, c.Board.Board().tiles[thingNeeded]))  
    elif action == commands.PlaceCityCommand:
        ctr.execute(action(player, thingNeeded, True))
    else:
        ctr.execute(action(player, thingNeeded))
    if(action == commands.BuyDevCardCommand):
        devCardsBought[player.id-1] += 1
    saveMove(save, player) 
    goNext()
    if(game.checkWon(player)):
        return
    if(action in commands.cardCommands()):
        turnCardUsed = True
    #La prima mossa è stata fatta
    goNext()
    while(action != commands.PassTurnCommand and not game.checkWon(player)):
        action, thingNeeded = game.bestAction(player, turnCardUsed)
        if action == commands.PlaceStreetCommand  or action == commands.PlaceColonyCommand:
            previousLongestStreetOwner = player.game.longestStreetPlayer(False)
            ctr.execute(action(player, thingNeeded, True))
            checkLongestStreetOwner(previousLongestStreetOwner, player)
        elif action == commands.UseKnightCommand:
            ctr.execute(action(player, thingNeeded))
            goNext()
            ctr.execute(commands.StealResourceCommand(player, c.Board.Board().tiles[thingNeeded]))
        elif action == commands.PlaceCityCommand:
            ctr.execute(action(player, thingNeeded, True))
        else:
            ctr.execute(action(player, thingNeeded))
        if(action == commands.BuyDevCardCommand):
            devCardsBought[player.id-1] += 1
        saveMove(save, player) 
        goNext()
        if(action in commands.cardCommands()):
            turnCardUsed = True

def playGameWithGraphic(game: c.GameWithCommands, view=None):
    global devCardsBought
    devCardsBought = [0.0, 0.0, 0.0, 0.0]
    GameView.GameView.setupAndDisplayBoard(view)
    GameView.GameView.setupPlaces(view)
    GameView.GameView.updateGameScreen(view)
    pygame.display.update()
    game.actualTurn = 0 
    won = False
    # START INIZIALE
    # game.players[0].AI = True
    # game.players[1].AI = True
    # game.players[2].AI = True
    # game.players[3].AI = True
    game.players[0].RANDOM = True
    game.players[1].RANDOM = True
    game.players[2].RANDOM = True
    game.players[3].RANDOM = True
    
    for p in game.players:
        game.doInitialChoise(p)
        GameView.GameView.updateGameScreen(view)
        saveMove(save, p) #################
        goNext()
    for p in sorted(game.players, reverse=True):
        game.doInitialChoise(p, giveResources = True)
        GameView.GameView.updateGameScreen(view)
        saveMove(save, p) #################
        goNext()

    while won == False:
        playerTurn = game.players[game.actualTurn%game.nplayer]
        game.currentTurnPlayer = playerTurn
        game.actualTurn += 1
        doTurnGraphic(game, playerTurn)
        if(playerTurn.victoryPoints >= 10):
            WINNERS[playerTurn.id-1] += 1
            s = 'Winner: ' + str(playerTurn.id) + "\n"
            game.printVictoryPointsOfAll(devCardsBought)
            saveToCsv(playerTurn)
            print(s) 
            won = True
    goNext()
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
    print(len(total))
    global allGames
    allGames = pd.concat([allGames, total], ignore_index=True)

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