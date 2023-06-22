from matplotlib import pyplot as plt
import pygame
import Classes as c
# from AI.Gnn import Gnn
import pandas as pd
import numpy as np
import csv
from Classes.Strategy.RLStrategyFFhier import ReinforcementLearningStrategyFfHier
from Classes.Strategy.RLStrategyGNN import ReinforcementLearningStrategyGnn
from Classes.Strategy.RLStrategyFF import ReinforcementLearningStrategyFf
import time
from Classes.Strategy.RLStrategyGNNhier import RLStrategyGnnHierarchical
from Classes.Strategy.EurPlayer import EuristicPlayer
from Classes.Strategy.RanPlayer import RandomPlayer
from Classes.Strategy.StrategyRandom import StrategyRandom
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
        rlStrategyFfHier = ReinforcementLearningStrategyFfHier()
        rEuristic = EuristicPlayer()
        strategies = [rlStrategyFfHier, rEuristic] #, rEuristic]
        withGraphics = False # True    
        idEpisode = 0
        gameCtrl = c.GameController.GameController(playerStrategies = strategies, idEpisode = idEpisode, withGraphics=withGraphics, speed=True)
        for seed in range(1, 11):
            winrates = [0,0]
            # print("Starting. Eps should be 1: ", .getEps()) # questo 
            print("Starting. Eps should be 1: ", rlStrategyFfHier.getEps()) # questo 
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
            # print("Definitely updated, final eps: ", .getEps(), flush = True)
            print("Definitely updated, final eps: ", rlStrategyFfHier.getEps(), flush = True)
            # rlStrategyGnn = ReinforcementLearningStrategyGnn()
            rlStrategyFfHier = ReinforcementLearningStrategyFfHier()

            strategies = [rlStrategyFfHier, rEuristic] # randomStrategy] #, rEuristic]
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

        


