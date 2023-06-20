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
from Classes.Strategy.RLStrategyGNNstreetSp import RLStrategyGnnStreet
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

if __name__ == '__main__':
        # prioStrategy = PriorityStrategy()
        # hybStrategy = HybridStrategy()
        # purStrategy = PureStrategy()
        rlStrategyGnn = ReinforcementLearningStrategyGnn()
        rlStrategyFf = ReinforcementLearningStrategyFf()
        rEuristic = RandomEuristicStrategy()
        rlSpecializedStreet = RLStrategyGnnStreet()
        # winners = []
        strategies = [rlStrategyFf, rEuristic]
        withGraphics = False # True #    
        idEpisode = 0
        gameCtrl = c.GameController.GameController(playerStrategies = strategies, idEpisode = idEpisode, withGraphics=withGraphics, speed=True)
        start_time = time.time()
        saveInCsv([strategies[0].name(), strategies[1].name()], "csvFolder/results.csv")
        for i in range(0, 100):
            print(".", end='')
            finalPoints = gameCtrl.playGameForTraining()
            saveInCsv(finalPoints, "csvFolder/results.csv")
            # idEpisode += 1
            gameCtrl.reset(idEpisode)
            if(i%100==0 and i > 0):
                 print(".")
            # plotWinners2(winners, strategies) 
        # if(withGraphics):
        #      pygame.quit()
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")
        # plt.savefig("plots/wPlot.png")
        # plotCsvColumnsWithHeaders("csvFolder/results.csv")

        


