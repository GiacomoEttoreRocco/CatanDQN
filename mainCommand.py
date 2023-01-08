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
import pygame_gui

speed = False
withGraphics = True
withDelay = False
realPlayer = False
save = False
train = False
ctr = controller.ActionController()

WINNERS = [0.0, 0.0, 0.0, 0.0]
devCardsBought = [0.0, 0.0, 0.0, 0.0]

def doActionWithGraphics(player):
    action, thingNeeded, onlyPassTurn = player.game.bestAction(player)
    ctr.execute(action(player, thingNeeded))
    view.updateGameScreen()
    if(not onlyPassTurn):  
        saveMove(save, player)
def undoActionWithGraphics():
    ctr.undo()
    view.updateGameScreen()
def redoActionWithGraphics():
    ctr.redo()
    view.updateGameScreen()


def decisionManager(player):
    assert not(not speed and not withGraphics)
    if(not speed and withGraphics):
        event = pygame.event.wait()
        # while event.type != pygame_gui.UI_BUTTON_PRESSED and event.type != pygame.KEYDOWN:
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if(event.ui_element == view.aiButton):
                player.AI = True
                player.RANDOM = False
                doActionWithGraphics(player)
            elif(event.ui_element == view.randomButton):
                player.AI = False
                player.RANDOM = True
                doActionWithGraphics(player)
            elif(event.ui_element == view.undoButton):
                undoActionWithGraphics()
            elif(event.ui_element == view.redoButton):
                redoActionWithGraphics()
            else:
                print("Nothing clicked")
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                print("Escape")
            elif event.key == pygame.K_a:
                player.AI = True
                player.RANDOM = False
                doActionWithGraphics(player)
            else:
                print(f'Key {event.key} pressed')
        if event.type == pygame.QUIT:
            print("Quitting")
            pygame.quit()

        view.manager.process_events(event) 
    
    elif (speed and withGraphics):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
        doActionWithGraphics(player)
        
    else:
        action, thingNeeded, onlyPassTurn = player.game.bestAction(player)
        ctr.execute(action(player, thingNeeded))
        if(not onlyPassTurn):  
            saveMove(save, player) 

    

# def doActionDecisions(game: c.GameWithCommands, player: c.PlayerWithCommands, withGraphics = True):
#     action = decisionManager(player)
    
#     if(action == commands.BuyDevCardCommand):
#         devCardsBought[player.id-1] += 1

#     if(game.checkWon(player)):
#         return
#     if(action in commands.cardCommands()):
#         player.turnCardUsed = True

def playGameWithGraphic(game: c.GameWithCommands, view=None, withGraphics = True):
    global devCardsBought
    devCardsBought = [0.0, 0.0, 0.0, 0.0]
    if(withGraphics):
        GameView.GameView.setupAndDisplayBoard(view)
        GameView.GameView.setupPlaces(view)
        GameView.GameView.updateGameScreen(view)
    game.actualTurn = 0 
    won = False
    game.players[0].AI = True
    game.players[1].AI = True
    game.players[2].AI = True
    game.players[3].AI = True
    # game.players[0].RANDOM = True
    # game.players[1].RANDOM = True
    # game.players[2].RANDOM = True
    # game.players[3].RANDOM = True
    
    reverseTurnOffSet = {0 : 0, 1 : 1, 2 : 2, 3 : 3, 4 : 3, 5 : 2, 6 : 1, 7 : 0}

    while won == False:
        if(game.actualTurn < 8):
            playerTurn = game.players[reverseTurnOffSet[game.actualTurn]]     
            decisionManager(playerTurn)
        else:
            playerTurn = game.players[game.actualTurn%game.nplayer]
            # doActionDecisions(game, playerTurn, withGraphics)
            decisionManager(playerTurn)
            if(playerTurn.victoryPoints >= 10):
                WINNERS[playerTurn.id-1] += 1
                s = 'Winner: ' + str(playerTurn.id) + "\n"
                # game.printVictoryPointsOfAll(devCardsBought)
                saveToJson(playerTurn)
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

def saveToJson(victoryPlayer):
    if(save):
        for i in range(len(total)):
            total.globals[i]['winner'] = victoryPlayer.id
        print("Length of total moves of this game: ", len(total))
        global allGames
        allGames = pd.concat([allGames, total], ignore_index=True)
        print("Length of total moves of allGames: ", len(allGames))


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
batchs = 100

for epoch in range(epochs):
    print('Iteration: ', epoch+1, "/", epochs)
    allGames = pd.DataFrame(data={'places': [], 'edges':[], 'globals':[]})   
    for batch in range(batchs): 
        print('game: ', batch+1, "/", batchs) 
        total = pd.DataFrame(data={'places': [], 'edges':[], 'globals':[]})
        #g = c.Game.Game()
        g = c.GameWithCommands.Game()
        if(withGraphics):
            view = GameView.GameView(g, ctr)
            playGameWithGraphic(g, view, withGraphics)
        else:
            playGameWithGraphic(g, None, withGraphics)
        c.Board.Board().reset()
        c.Bank.Bank().reset()

    if(train):
        allGames.to_json("./json/game.json")
        Gnn.Gnn().trainModel()
        printWinners()