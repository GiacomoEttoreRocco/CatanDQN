import pygame
import pygame_gui
import pandas as pd
from Classes import Board
from Classes.Strategy.HybridStrategy import HybridStrategy
from Classes.Strategy.PriorityStrategy import PriorityStrategy
from Classes.Strategy.PureStrategy import PureStrategy
from Classes.Strategy.ReinforcementLearningStrategy import ReinforcementLearningStrategy
from Classes.Strategy.Strategy import Strategy
from Command import commands
import Graphics.GameView as GameView
import AI.Gnn as Gnn
import Classes as c

from matplotlib import pyplot as plt


class GameController:

    def __init__(self, playerStrategies, speed=True, saveOnFile=True, withGraphics=False) -> None:
        self.speed = speed
        self.saveOnFile =saveOnFile
        self.withGraphics = withGraphics

        self.reset()
        self.game = c.Game.Game(len(playerStrategies)) 

        for player, strategy in zip(self.game.players, playerStrategies):
            player.strategy = strategy

        if self.withGraphics:
            self.view = GameView.GameView(self.game, self.game.ctr)
        else:
            self.view = None
        
        self.total = pd.DataFrame(data={'places': [], 'edges':[], 'globals':[]})

    def reset(self):
        c.Board.Board().reset()
        c.Bank.Bank().reset()
        Gnn.Gnn().reset()

    def executeWithDeltaReward(self, player, action, thingNeeded, onlyPassTurn):
        prevPoints = player._victoryPoints
        if(player.strategy.name() == "RL" and not onlyPassTurn): # action != commands.PassTurnCommand):
            previousGraph = Board.Board().boardStateGraph(player)
            previousGlob = player.globalFeaturesToTensor()
            actionId = player.strategy.getActionId(action)

        self.game.ctr.execute(action(player, thingNeeded))
        player.reward = player._victoryPoints - prevPoints

        if(player.strategy.name() == "RL" and not onlyPassTurn): # action != commands.PassTurnCommand):
            graph = Board.Board().boardStateGraph(player)
            glob = player.globalFeaturesToTensor()
            # print("Linea 54 GameController, actionIs: ", actionId.value)
            if(actionId.value > 0):
                print("Move saved in memory.")
                player.strategy.macroDQN.saveInMemory(previousGraph, previousGlob, actionId.value, player.reward, graph, glob)

    def decisionManagerGUI(self, player):
        if(not self.speed and self.withGraphics):
            event = pygame.event.wait()
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if(event.ui_element == self.view.pureButton):
                    player.strategy = PureStrategy()
                elif(event.ui_element == self.view.priorityButton):
                    player.strategy = PriorityStrategy()
                elif(event.ui_element == self.view.hybridButton):
                    player.strategy = HybridStrategy()
                elif(event.ui_element == self.view.rlButton):
                    player.strategy = ReinforcementLearningStrategy()
                elif(event.ui_element == self.view.undoButton):
                    self.game.ctr.undo()
                elif(event.ui_element == self.view.redoButton):
                    self.game.ctr.redo()
                elif(event.ui_element == self.view.doButton):
                    action, thingNeeded, onlyPassTurn = player.bestAction()
                    # self.game.ctr.execute(action(player, thingNeeded))
                    self.executeWithDeltaReward(player, action, thingNeeded, onlyPassTurn) 
                elif(event.ui_element == self.view.stack.scroll_bar.bottom_button):
                    self.view.stack.scroll_bar.set_scroll_from_start_percentage(self.view.stack.scroll_bar.start_percentage+0.1)
                    # self.view.updateGameScreen(True)
                elif(event.ui_element == self.view.stack.scroll_bar.top_button):
                    self.view.stack.scroll_bar.set_scroll_from_start_percentage(self.view.stack.scroll_bar.start_percentage-0.1)
                    # self.view.updateGameScreen(True)
                else:
                    print("Nothing clicked")
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    print("Escape")
            # if event.type == pygame.QUIT:
                    print("Quitting")
                    pygame.quit()
            self.view.updateGameScreen()
            self.view.manager.process_events(event) 
        else:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
            # print("Player type: ", player.strategy.name())
            action, thingNeeded, onlyPassTurn = player.bestAction()
            # print("Action: ", action, "thingNeeded: ", thingNeeded, "OnlyPassTurn: ", onlyPassTurn)
            # self.game.ctr.execute(action(player, thingNeeded))
            self.executeWithDeltaReward(player, action, thingNeeded, onlyPassTurn)

            # if(not onlyPassTurn):  
            #     self.saveMove(player) 

    def decisionManagerNonGUI(self, player):
        action, thingNeeded, onlyPassTurn = player.bestAction()
        # self.game.ctr.execute(action(player, thingNeeded))
        self.executeWithDeltaReward(player, action, thingNeeded, onlyPassTurn)
        # self.plotVictoryPoints(player._victoryPoints, player.id)

        # if(not onlyPassTurn):  
        #     # self.saveMove(player) 

    def decisionManager(self, player):
        if(self.withGraphics):
            self.decisionManagerGUI(player)
            self.view.updateGameScreen()
        else:
            self.decisionManagerNonGUI(player)

    def playGame(self):    
        self.reset()
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
                if(playerTurn._victoryPoints >= 10):
                    # self.saveToJson(playerTurn)
                    print(f'Winner: {playerTurn.id}, Agent: {playerTurn.strategy.name()}\n')
                    if(self.withGraphics):
                        pygame.quit()
                    return playerTurn
                
            self.plotVictoryPoints(playerTurn._victoryPoints, playerTurn.id)
            

    # def saveMove(self, player):
    #     if(self.saveOnFile):
    #         places = c.Board.Board().placesToDict(player)
    #         edges = c.Board.Board().edgesToDict(player)
    #         globals = player.globalFeaturesToDict()
    #         self.total.loc[len(self.total)] = [places, edges, globals]

    # def saveToJson(self, victoryPlayer):
    #     if(self.saveOnFile):
    #         for i in range(len(self.total)):
    #             self.total.globals[i]['winner'] = 1 if self.total.globals[i]['player_id'] == victoryPlayer.id else 0
    #             del self.total.globals[i]['player_id']
    #         print("Length of total moves of this game: ", len(self.total))

    valueFunction1 = []
    valueFunction2 = []

    def resetPlot(self):
        self.valueFunction1 = []
        self.valueFunction2 = []
        plt.clf()

    def plotVictoryPoints(self, points, id):
        plt.figure(1)
        
        plt.xlabel('Turns')
        plt.ylabel('Vicotry points')
        
        if id == 1:
            self.valueFunction1.append(points)
            plt.plot(self.valueFunction1, color='red', label='Player SL')
        elif id == 2:
            self.valueFunction2.append(points)
            plt.plot(self.valueFunction2, color='orange', label='Player RL')
        

        handles, labels = plt.gca().get_legend_handles_labels()
        if len(handles) == 2:
            plt.legend()
        # plt.tight_layout()
        plt.pause(0.001)

