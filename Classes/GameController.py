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
        action, thingNeeded, onlyPassTurn = player.game.bestAction(player)
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

    
    def decisionManager(self, player):
        assert not(not self.speed and not self.withGraphics)
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
                elif event.key == pygame.K_a:
                    player.type = PlayerTypes.PURE
                    self.doActionWithGraphics(player)
                else:
                    print(f'Key {event.key} pressed')
            if event.type == pygame.QUIT:
                print("Quitting")
                pygame.quit()

            self.view.manager.process_events(event) 
        
        elif (self.speed and self.withGraphics):
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
            self.doActionWithGraphics(player)
            
        else:
            action, thingNeeded, onlyPassTurn = player.game.bestAction(player)
            self.game.ctr.execute(action(player, thingNeeded))
            if(not onlyPassTurn):  
                self.saveMove(player) 
                    
            
    def playGameWithGraphic(self):
        
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
            else:
                playerTurn = self.game.players[self.game.actualTurn%self.game.nplayers]
                self.decisionManager(playerTurn)
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

