import pygame
import pygame_gui
import pandas as pd
import Graphics.GameView as GameView
import AI.Gnn as Gnn
import Classes as c
from Classes.PlayerTypes import PlayerTypes

class GameController:

    def __init__(self, playerTypes: list(PlayerTypes), speed=True, saveOnFile=True, withGraphics=False) -> None:
        self.speed = speed
        self.saveOnFile =saveOnFile
        self.withGraphics = withGraphics

        self.reset()
        self.game = c.Game.Game(len(playerTypes)) 

        for player, type in zip(self.game.players, playerTypes):
            player.type = type

        if self.withGraphics:
            self.view = GameView.GameView(self.game, self.game.ctr)
        else:
            self.view = None
        
        self.total = pd.DataFrame(data={'places': [], 'edges':[], 'globals':[]})

    def reset(self):
        c.Board.Board().reset()
        c.Bank.Bank().reset()
        Gnn.Gnn().reset()

    def doActionWithGraphics(self, player):
        action, thingNeeded, onlyPassTurn = player.bestActionSL()
        self.game.ctr.execute(action(player, thingNeeded))
        self.view.updateGameScreen()
        if(not onlyPassTurn):  
            self.saveMove(player)

    def undoActionWithGraphics(self):
        self.game.ctr.undo()
        self.view.updateGameScreen()

    def redoActionWithGraphics(self):
        self.game.ctr.redo()
        self.view.updateGameScreen()    

    def decisionManagerGUI(self, player):
        if(not self.speed and self.withGraphics):
            event = pygame.event.wait()
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if(event.ui_element == self.view.aiButton):
                    player.type = PlayerTypes.PURE
                    self.doActionWithGraphics(player)
                elif(event.ui_element == self.view.randomButton):
                    player.type = PlayerTypes.PRIORITY
                    self.doActionWithGraphics(player)
                elif(event.ui_element == self.view.undoButton):
                    self.undoActionWithGraphics()
                elif(event.ui_element == self.view.redoButton):
                    self.redoActionWithGraphics()
                elif(event.ui_element == self.view.stack.scroll_bar.bottom_button):
                    self.view.stack.scroll_bar.set_scroll_from_start_percentage(self.view.stack.scroll_bar.start_percentage+0.1)
                    self.view.updateGameScreen(True)
                elif(event.ui_element == self.view.stack.scroll_bar.top_button):
                    self.view.stack.scroll_bar.set_scroll_from_start_percentage(self.view.stack.scroll_bar.start_percentage-0.1)
                    self.view.updateGameScreen(True)
                else:
                    print("Nothing clicked")
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    print("Escape")
            if event.type == pygame.QUIT:
                print("Quitting")
                pygame.quit()
            self.view.manager.process_events(event) 
        else:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
            self.doActionWithGraphics(player)

    def decisionManagerNonGUI(self, player):
        action, thingNeeded, onlyPassTurn = player.bestActionSL()
        self.game.ctr.execute(action(player, thingNeeded))
        if(not onlyPassTurn):  
            self.saveMove(player) 

    def decisionManager(self, player):
        if(self.withGraphics):
            self.decisionManagerGUI(player)
        else:
            self.decisionManagerNonGUI(player)

#############################################################################################################

    def decisionManager_RL(self, player):
        if(self.withGraphics):
            self.decisionManagerGUI_RL(player)
        else:
            self.decisionManagerNonGUI_RL(player)

    def decisionManagerNonGUI_RL(self, player):
        # action, thingNeeded, onlyPassTurn = player.bestActionSL()
        state = self.game.getState(player)
        action, thingNeeded, onlyPassTurn = player.bestActionRL(state)
        action = action(player, thingNeeded)
        # self.game.ctr.execute(action(player, thingNeeded))
        reward = self.game.step(action(player, thingNeeded))
        newState = self.game.getState(player)

#############################################################################################################

    def playGame(self):    
        if(self.withGraphics):
            GameView.GameView.setupAndDisplayBoard(self.view)
            GameView.GameView.setupPlaces(self.view)
            GameView.GameView.updateGameScreen(self.view)
                 
        self.game.actualTurn = 0       
        reverseTurnOffSet = [*list(range(self.game.nplayers)), *list(reversed(range(self.game.nplayers)))]
        while True:
            if(self.game.actualTurn < self.game.nplayers*2):
                playerTurn = self.game.players[reverseTurnOffSet[self.game.actualTurn]] 
                self.decisionManager(playerTurn)
                # print("Tensor state initial phase: \n")    
                # print("Board state: ", self.game.getBoardState(playerTurn))
                # print("Player state: ", self.game.getPlayerGlobalFeaturesState(playerTurn))
            else:
                playerTurn = self.game.players[self.game.actualTurn%self.game.nplayers]
                self.decisionManager(playerTurn)
                # print("Generic turn: \n")
                # print("Board state: ", self.game.getBoardState(playerTurn))
                # print("Player state: ", self.game.getPlayerGlobalFeaturesState(playerTurn))

                if(playerTurn.victoryPoints >= 10):
                    self.saveToJson(playerTurn)
                    print(f'Winner: {playerTurn.id}, Agent: {playerTurn.type}\n')
                    if(self.withGraphics):
                        pygame.quit()
                    
                    return playerTurn

    def saveMove(self, player):
        if(self.saveOnFile):
            places = c.Board.Board().placesToDict(player)
            edges = c.Board.Board().edgesToDict(player)
            globals = player.globalFeaturesToDict()
            self.total.loc[len(self.total)] = [places, edges, globals]

    def saveToJson(self, victoryPlayer):
        if(self.saveOnFile):
            for i in range(len(self.total)):
                self.total.globals[i]['winner'] = 1 if self.total.globals[i]['player_id'] == victoryPlayer.id else 0
                del self.total.globals[i]['player_id']
            print("Length of total moves of this game: ", len(self.total))
