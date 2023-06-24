import os
import pygame
import pygame_gui
import pandas as pd
from Classes import Bank, Board
# from Classes.Strategy.HybridStrategy import HybridStrategy
# from Classes.Strategy.PriorityStrategy import PriorityStrategy
# from Classes.Strategy.PureStrategy import PureStrategy
from Classes.Strategy.RLStrategyGNN import ReinforcementLearningStrategyGnn
from Classes.Strategy.RLStrategyFF import ReinforcementLearningStrategyFf

from Classes.Strategy.Strategy import Strategy
from Classes.staticUtilities import tradesToId
from Command import commands
import Graphics.GameView as GameView
import AI.Gnn as Gnn
import Classes as c

from matplotlib import pyplot as plt

class GameController:

    def __init__(self, playerStrategies, idEpisode, speed=True, withGraphics=False) -> None:
        self.prelimit = 10 # episodes before plots
        self.idEpisode = idEpisode 

        self.speed = speed
        self.withGraphics = withGraphics
        self.playerStrategies = playerStrategies
        self.game = c.Game.Game(len(playerStrategies)) 

        for player, strategy in zip(self.game.players, self.playerStrategies):
            # print("Riga 32 gamecontroller:", strategy)
            player.strategy = strategy

        if self.withGraphics:
            self.view = GameView.GameView(self.game, self.game.ctr)
        else:
            self.view = None
        
        self.total = pd.DataFrame(data={'places': [], 'edges':[], 'globals':[]})

    def reset(self): #, newStrat):
        c.Board.Board().reset()
        c.Bank.Bank().reset()
        # Gnn.Gnn().reset()
        self.game.reset()
        # print("GLOBAL RESET")
        # self.idEpisode = idEpisode
        # if(idEpisode > self.prelimit):
        #     self.resetPlot()
        # self.playerStrategies = newStrat
        for player, strategy in zip(self.game.players, self.playerStrategies):
            # print("Riga 32 gamecontroller:", strategy)
            player.strategy = strategy
            player.reset()

    # def executeWithDeltaReward(self, player, action, thingNeeded, onlyPassTurn):
    #     prevPoints = player._victoryPoints

    #     rlFlag = "RL" in player.strategy.name() and not onlyPassTurn

    #     if(rlFlag): 
    #         if("GNN" in player.strategy.name()):
    #             previousGraph = Board.Board().boardStateGraph(player)
    #             previousGlob = player.globalFeaturesToTensor()
    #             actionId = player.strategy.getActionId(action)
    #         else:
    #             previousState = self.game.getTotalState(player)
    #             actionId = player.strategy.getActionId(action)

    #     self.game.ctr.execute(action(player, thingNeeded))
    #     player.reward = player._victoryPoints - prevPoints

    #     if(player._victoryPoints >= 10):
    #         print("winning reward added")
    #         player.reward = 100

    #     if(rlFlag): 
    #         graph = Board.Board().boardStateGraph(player)
    #         glob = player.globalFeaturesToTensor()
    #         # print("Riga 72, game controller: ", player.reward)

    #         if(actionId.value > 0 and "GNN" in player.strategy.name()):
    #             player.strategy.macroDQN.saveInMemory(previousGraph, previousGlob, actionId.value, player._victoryPoints, graph, glob)
    #             if("HIER" in player.strategy.name()):
    #                 if(actionId.value == 2):
    #                     player.strategy.streetDQN.saveInMemory(previousGraph, previousGlob, list(Board.Board().edges.keys()).index(thingNeeded), player._victoryPoints, graph, glob)
    #                 if(actionId.value == 3):
    #                     player.strategy.colonyDQN.saveInMemory(previousGraph, previousGlob, Board.Board().places.index(thingNeeded), player._victoryPoints, graph, glob)
    #                 if(actionId.value == 5): 
    #                     player.strategy.tradeDQN.saveInMemory(previousGraph, previousGlob, tradesToId(thingNeeded), player._victoryPoints, graph, glob)
                       
    #         elif(actionId.value > 0):
    #             player.strategy.macroDQN.saveInMemory(previousState, actionId.value, player._victoryPoints, self.game.getTotalState(player))
    #             if("HIER" in player.strategy.name()):
    #                 if(actionId.value == 2):
    #                     player.strategy.streetDQN.saveInMemory(previousState, list(Board.Board().edges.keys()).index(thingNeeded), player._victoryPoints, self.game.getTotalState(player))
    #                 if(actionId.value == 3):
    #                     player.strategy.colonyDQN.saveInMemory(previousState, Board.Board().places.index(thingNeeded), player._victoryPoints, self.game.getTotalState(player))
    #                 if(actionId.value == 5): 
    #                     player.strategy.tradeDQN.saveInMemory(previousState, tradesToId(thingNeeded), player._victoryPoints, self.game.getTotalState(player))

    def executeWithDeltaReward(self, player, action, thingNeeded, onlyPassTurn):
        # prevPoints = player._victoryPoints

        rlFlag = "RL" in player.strategy.name() and not onlyPassTurn

        if(rlFlag): 
            if("GNN" in player.strategy.name()):
                previousGraph = Board.Board().boardStateGraph(player)
                previousGlob = player.globalFeaturesToTensor()
                actionId = player.strategy.getActionId(action)
            else:
                previousState = self.game.getTotalState(player)
                actionId = player.strategy.getActionId(action)

        self.game.ctr.execute(action(player, thingNeeded))
        # player.reward = player._victoryPoints - prevPoints
        player.reward = player._victoryPoints/self.game.actualTurn

        if(player._victoryPoints >= 10):
            # print("winning reward added")
            player.reward = 100

        if(player._victoryPoints == 2 and self.game.actualTurn > 50):
            # print("penality reward added")
            player.reward = -1

        if(rlFlag): 
            graph = Board.Board().boardStateGraph(player)
            glob = player.globalFeaturesToTensor()
            # print("Riga 72, game controller: ", player.reward)

            if(actionId.value > 0 and "GNN" in player.strategy.name()):
                player.strategy.macroDQN.saveInMemory(previousGraph, previousGlob, actionId.value, player.reward, graph, glob)
                if("HIER" in player.strategy.name()):
                    if(actionId.value == 2 or actionId.value == -6 or actionId.value == 8):
                        if(actionId.value != 2):
                            print("Riga 140 GameCtrl. Hier, fai un fischio", actionId.value)
                        player.strategy.streetDQN.saveInMemory(previousGraph, previousGlob, list(Board.Board().edges.keys()).index(thingNeeded), player.reward, graph, glob)
                    if(actionId.value == 3 or actionId.value == -3 or actionId.value == -2):
                        if(actionId.value != 3):
                            print("Riga 144 GameCtrl, Hier, fai un fischio", actionId.value)
                        player.strategy.colonyDQN.saveInMemory(previousGraph, previousGlob, Board.Board().places.index(thingNeeded), player.reward, graph, glob)
                    if(actionId.value == 5): 
                        player.strategy.tradeDQN.saveInMemory(previousGraph, previousGlob, tradesToId(thingNeeded), player.reward, graph, glob)
                       
            elif(actionId.value > 0):
                player.strategy.macroDQN.saveInMemory(previousState, actionId.value, player.reward, self.game.getTotalState(player))
                if("HIER" in player.strategy.name()):
                    if(actionId.value == 2 or actionId.value == -6 or actionId.value == 8):
                        if(actionId.value != 2):
                            print("Riga 154 GameCtrl, macro, fai un fischio", actionId.value)
                        player.strategy.streetDQN.saveInMemory(previousState, list(Board.Board().edges.keys()).index(thingNeeded), player.reward, self.game.getTotalState(player))
                    if(actionId.value == 3 or actionId.value == -3 or actionId.value == -2):
                        if(actionId.value != 3):
                            print("Riga 154 GameCtrl, macro, fai un fischio", actionId.value)
                        player.strategy.colonyDQN.saveInMemory(previousState, Board.Board().places.index(thingNeeded), player.reward, self.game.getTotalState(player))
                    if(actionId.value == 5): 
                        player.strategy.tradeDQN.saveInMemory(previousState, tradesToId(thingNeeded), player.reward, self.game.getTotalState(player))

                
    def decisionManagerGUI(self, player):
        if(not self.speed and self.withGraphics):
            event = pygame.event.wait()
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                # if(event.ui_element == self.view.pureButton):
                #     player.strategy = PureStrategy()
                # elif(event.ui_element == self.view.priorityButton):
                #     player.strategy = PriorityStrategy()
                # elif(event.ui_element == self.view.hybridButton):
                #     player.strategy = HybridStrategy()
                # elif(event.ui_element == self.view.rlButton):
                #     player.strategy = ReinforcementLearningStrategy()
                if(event.ui_element == self.view.undoButton):
                    self.game.ctr.undo()
                elif(event.ui_element == self.view.redoButton):
                    self.game.ctr.redo()
                elif(event.ui_element == self.view.doButton):
                    action, thingNeeded, onlyPassTurn = player.bestAction()
                    print("Riga 123 game controller: ", action)
                    self.executeWithDeltaReward(player, action, thingNeeded, onlyPassTurn) 
                elif(event.ui_element == self.view.stack.scroll_bar.bottom_button):
                    self.view.stack.scroll_bar.set_scroll_from_start_percentage(self.view.stack.scroll_bar.start_percentage+0.1)
                elif(event.ui_element == self.view.stack.scroll_bar.top_button):
                    self.view.stack.scroll_bar.set_scroll_from_start_percentage(self.view.stack.scroll_bar.start_percentage-0.1)
                else:
                    print("Nothing clicked")
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    print("Escape")
                    print("Quitting")
            self.view.updateGameScreen()
            self.view.manager.process_events(event) 
        else:
            events = pygame.event.get()
            action, thingNeeded, onlyPassTurn = player.bestAction()
            self.executeWithDeltaReward(player, action, thingNeeded, onlyPassTurn)


    def decisionManagerNonGUI(self, player):
        action, thingNeeded, onlyPassTurn = player.bestAction()
        self.executeWithDeltaReward(player, action, thingNeeded, onlyPassTurn)

    def decisionManager(self, player):
        if(self.withGraphics):
            self.decisionManagerGUI(player)
            self.view.updateGameScreen()
        else:
            self.decisionManagerNonGUI(player)

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
                if(playerTurn._victoryPoints >= 10): # max length
                    print(f'Winner: {playerTurn.id}, Agent: {playerTurn.strategy.name()}\n')
                    # self.plotVictoryPoints(playerTurn._victoryPoints, playerTurn.id)
                    return playerTurn
                if(self.game.actualTurn > 1000):
                    return playerTurn
            # self.plotVictoryPoints(playerTurn._victoryPoints, playerTurn.id)
            #################################################################
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
    lastId = 2
    # idEpisode = 0

    def playGameForTraining(self):    
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
                if(playerTurn._victoryPoints >= 10 or self.game.actualTurn >= 120): 
                    # print(f'Winner: {playerTurn.id}, Agent: {playerTurn.strategy.name()}\n')
                    toReturn = []
                    for player in self.game.players:
                        toReturn.append(player._victoryPoints)
                        if("RL" in player.strategy.name()):
                            player.strategy.epsDecay()
                            # print(player.strategy.getEps())
                    # print(toReturn)
                    return toReturn

                
    def playTurnamentGame(self):    
        if(self.withGraphics):
            GameView.GameView.setupAndDisplayBoard(self.view)
            GameView.GameView.setupPlaces(self.view)
            GameView.GameView.updateGameScreen(self.view)
        self.game.actualTurn = 0       
        reverseTurnOffSet = [*list(range(self.game.nplayers)), *list(reversed(range(self.game.nplayers)))]
        pointsAt100 = []
        pointsAtFinal = []
        saved = False
        while True:
            if(self.game.actualTurn > 1000):
                return "DUMMY", pointsAt100, pointsAtFinal
            if(self.game.actualTurn < self.game.nplayers*2):
                playerTurn = self.game.players[reverseTurnOffSet[self.game.actualTurn]] 
                self.decisionManager(playerTurn)
            else:
                playerTurn = self.game.players[self.game.actualTurn%self.game.nplayers]
                self.decisionManager(playerTurn)
                if(self.game.actualTurn == 121 and not saved): 
                    saved = True
                    for player in self.game.players:
                        pointsAt100.append(player._victoryPoints)
                if(playerTurn._victoryPoints >= 10):
                    for player in self.game.players:
                        pointsAtFinal.append(player._victoryPoints)
                    return playerTurn.strategy.name(), pointsAt100, pointsAtFinal

    def resetPlot(self):
        self.valueFunction1 = []
        self.valueFunction2 = []
        # plt.savefig("plots/episode"+ str(self.idEpisode)+".png")
        if not os.path.exists("plots"):
            os.makedirs("plots")
        plt.savefig("plots/episode{}.png".format(self.idEpisode))
        plt.clf()
        # self.idEpisode = self.idEpisode + 1
        print("Number of episode: ", self.idEpisode)

    def plotVictoryPoints(self, points, idPlayer):
        prelimit = self.prelimit
        if(self.idEpisode > prelimit):
            if(idPlayer != self.lastId):
                plt.figure(1)
                plt.xlabel('Turns')
                plt.ylabel('Victory points')
                if idPlayer == 1:
                    self.valueFunction1.append(points)
                    plt.plot(self.valueFunction1, color='red', label='Player ' + self.game.players[0].strategy.name())
                elif idPlayer == 2:
                    self.valueFunction2.append(points)
                    plt.plot(self.valueFunction2, color='orange', label='Player ' + self.game.players[1].strategy.name())
                    handles, labels = plt.gca().get_legend_handles_labels()
                    if len(handles) < 3:
                        plt.legend()
                    # plt.tight_layout()
                    plt.title("Episode "+ str(self.idEpisode))
                    plt.pause(0.001)
                self.lastId = idPlayer

