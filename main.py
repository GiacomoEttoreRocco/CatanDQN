from matplotlib import pyplot as plt
import pygame
import Classes as c
# from AI.Gnn import Gnn
import pandas as pd
import numpy as np
import csv
# from Classes.Strategy.HybridStrategy import HybridStrategy
# from Classes.Strategy.PriorityStrategy import PriorityStrategy
# from Classes.Strategy.PureStrategy import PureStrategy
from Classes.Strategy.RLStrategyGNN import ReinforcementLearningStrategyGnn
from Classes.Strategy.RLStrategyFF import ReinforcementLearningStrategyFf
import time
from Classes.Strategy.RLStrategyGNNstreetSp import RLStrategyGnnHierarchical
from Classes.Strategy.RandomEuristic import RandomEuristicStrategy
from Classes.staticUtilities import plotCsvColumns, plotCsvColumnsWithHeaders, plotWinners2, saveInCsv

def training(playerStrategies, iterationProcessIndex, iterations, numberOfTrainingGames, numberOfValidationGames):
    winners = [0.0] * len(playerStrategies)
    print(f'Starting training: {iterationProcessIndex}')
    for iteration in range(iterations):
        print('Iteration: ', iteration+1, "/", iterations)
        allGames = pd.DataFrame(data={'places': [], 'edges':[], 'globals':[]})
        np.random.shuffle(playerStrategies)
        for numGame in range(numberOfTrainingGames): 
            print('game: ', numGame+1, "/", numberOfTrainingGames) 
            game = c.GameController.GameController(playerStrategies=playerStrategies)
            winner = game.playGameWithGraphic()
            winners[winner.id-1]+=1
            allGames = pd.concat([allGames, game.total], ignore_index=True)
        print("Length of total moves of allGames: ", len(allGames))
        # printWinners(winners)
        allGames.to_json("./json/training_game.json")
        allGames = pd.DataFrame(data={'places': [], 'edges':[], 'globals':[]})   
        for numGame in range(numberOfValidationGames): 
            print('game: ', numGame+1, "/", numberOfValidationGames) 
            game = c.GameController.GameController(playerStrategies=playerStrategies)
            winner = game.playGameWithGraphic()
            winners[winner.id-1]+=1
            allGames = pd.concat([allGames, game.total], ignore_index=True)

        print("Length of total moves of allGames: ", len(allGames))
        allGames.to_json("./json/testing_game.json")

def writeOnCsv(i, winners):
    with open('results.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([i, *winners])

def compareHierVsEur():
        rEuristic = RandomEuristicStrategy()
        rlHier = RLStrategyGnnHierarchical()
        strategies = [rlHier, rEuristic]
        withGraphics = False    
        idEpisode = 0
        gameCtrl = c.GameController.GameController(playerStrategies = strategies, idEpisode = idEpisode, withGraphics=withGraphics, speed=True)
        for seed in range(10, 11):
            winrates = [0,0]
            print("Starting. Eps should be 1: ", rlHier.getEps())
            saveInCsv([strategies[0].name(), strategies[1].name()], "csvFolder/results"+str(seed)+".csv")
            for i in range(0, 200):
                print(".", end='', flush = True)
                finalPoints = gameCtrl.playGameForTraining()
                saveInCsv(finalPoints, "csvFolder/results"+str(seed)+".csv")
                if(finalPoints[0] > finalPoints[1]):
                    winrates[0]+=1
                else:
                    winrates[1]+=1
                gameCtrl.reset(strategies)
            print("Winrates: ", winrates)
            print("Definitely updated, final eps: ", rlHier.getEps(), flush = True)
            rlHier = RLStrategyGnnHierarchical()
            strategies = [rlHier, rEuristic]
            gameCtrl.reset(strategies)

def compare():
        rlStrategyGnn = ReinforcementLearningStrategyGnn()
        rlStrategyFf = ReinforcementLearningStrategyFf()
        rEuristic = RandomEuristicStrategy()
        rlHier = RLStrategyGnnHierarchical()
        strategies = [rlStrategyFf, rEuristic]
        withGraphics = False    
        idEpisode = 0
        gameCtrl = c.GameController.GameController(playerStrategies = strategies, idEpisode = idEpisode, withGraphics=withGraphics, speed=True)
        for seed in range(1, 10):
            winrates = [0,0]
            print("Starting. Eps should be 1: ", rlHier.getEps())
            saveInCsv([strategies[0].name(), strategies[1].name()], "csvFolder/results"+str(seed)+".csv")
            for i in range(0, 200):
                print(".", end='', flush = True)
                finalPoints = gameCtrl.playGameForTraining()
                saveInCsv(finalPoints, "csvFolder/results"+str(seed)+".csv")
                if(finalPoints[0] > finalPoints[1]):
                    winrates[0]+=1
                else:
                    winrates[1]+=1
                gameCtrl.reset(strategies)
            print("Winrates: ", winrates)
            print("Definitely updated, final eps: ", rlHier.getEps(), flush = True)
            rlHier = RLStrategyGnnHierarchical()
            strategies = [rlHier, rEuristic]
            gameCtrl.reset(strategies)

if __name__ == '__main__':
        # prioStrategy = PriorityStrategy()
        # hybStrategy = HybridStrategy()
        # purStrategy = PureStrategy()
        # rlHier = RLStrategyGnnHierarchical()
        # strategies = [rlSpecializedStreet, rEuristic]
        # strategies = [rEuristic, rEuristic]
        # rlStrategyFf = ReinforcementLearningStrategyFf()

        rlStrategyGnn = ReinforcementLearningStrategyGnn()
        rEuristic = RandomEuristicStrategy()

        strategies = [rlStrategyGnn, rEuristic]

        # withGraphics = True 
        withGraphics = False #    
        idEpisode = 0
        gameCtrl = c.GameController.GameController(playerStrategies = strategies, idEpisode = idEpisode, withGraphics=withGraphics, speed=True)
        #####################################################################################
        #####################################################################################
        seed = 2
    
        for seed in range(1, 11):
            winrates = [0,0]
            print("Starting. Eps should be 1: ", rlStrategyGnn.getEps()) # questo
            saveInCsv([strategies[0].name(), strategies[1].name()], "csvFolder/results"+str(seed)+".csv")
            for i in range(0, 200):
                print(".", end='', flush = True)
                finalPoints = gameCtrl.playGameForTraining()
                saveInCsv(finalPoints, "csvFolder/results"+str(seed)+".csv")
                if(finalPoints[0] > finalPoints[1]):
                    winrates[0]+=1
                else:
                    winrates[1]+=1
                gameCtrl.reset(strategies)
            print("Winrates: ", winrates)
            print("Definitely updated, final eps: ", rlStrategyGnn.getEps(), flush = True)
            rlStrategyGnn = ReinforcementLearningStrategyGnn()
            strategies = [rlStrategyGnn, rEuristic]
            gameCtrl.reset(strategies)
            
        #####################################################################################
        #####################################################################################
            # plotWinners2(winners, strategies) 
        # if(withGraphics):
        #      pygame.quit()
        # end_time = time.time()
        # execution_time = end_time - start_time
        # print(f"Execution time: {execution_time} seconds")
        # plt.savefig("plots/wPlot.png")
        # plotCsvColumnsWithHeaders("csvFolder/results.csv")

        


