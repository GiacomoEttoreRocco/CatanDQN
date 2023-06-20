from matplotlib import pyplot as plt
import pygame
import Classes as c
from AI.Gnn import Gnn
import pandas as pd
import numpy as np
import csv
from Classes.Strategy.HybridStrategy import HybridStrategy
from Classes.Strategy.PriorityStrategy import PriorityStrategy
from Classes.Strategy.PureStrategy import PureStrategy
from Classes.Strategy.RLStrategyGNN import ReinforcementLearningStrategyGnn
from Classes.Strategy.RLStrategyFF import ReinforcementLearningStrategyFf
import time
from Classes.Strategy.RLStrategyGNNstreetSp import RLStrategyGnnStreet

from Classes.Strategy.RandomEuristic import RandomEuristicStrategy
from Classes.staticUtilities import plotWinners2

def printWinners(winners):
        normValue = sum(winners)
        toPrint = [0.0, 0.0, 0.0, 0.0]
        for i, v in enumerate(winners):
            toPrint[i] = v/normValue
        s = ""
        for i, vperc in enumerate(toPrint):
            s = s + "Player " + str(i+1)+ ": " + str(round(vperc*100.0,2)) + " % "
        print(s)
        print(winners)

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
        # printWinners(winners)
        allGames.to_json("./json/testing_game.json")
        
        Gnn().trainModel(validate=True)

def performanceEvaluation(iterationProcessIndex, playerTypes, numberOfTestingGames, withGraphics, speed):
    print("PERFORMANCE EVALUATION STARTED...")
    winners = [0.0, 0.0]
    special = playerTypes[-1]
    for numGame in range(numberOfTestingGames): 
        np.random.shuffle(playerTypes)
        print('game: ', numGame+1, "/", numberOfTestingGames) 
        game = c.GameController.GameController(playerTypes=playerTypes, withGraphics=withGraphics, speed=speed, saveOnFile=False)
        winner = game.playGameWithGraphic()
        if winner.type == special:
            winners[0]+=1
        else:
            winners[1]+=1
    
    print(f'PERFORMANCE EVALUATION FINISHED. RESULT: {winners[0]/sum(winners)*100.0} %') 
    return winners[0]/sum(winners)*100.0

    
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

        winners = []
        # strategies = [rlStrategyGnn, rEuristic, rlStrategyFf]
        # strategies = [rEuristic, rlSpecializedStreet]
        strategies = [rlStrategyGnn, rEuristic]
        withGraphics = False # True #   
        idEpisode = 0
        gameCtrl = c.GameController.GameController(playerStrategies = strategies, idEpisode = idEpisode, withGraphics=withGraphics, speed=True, saveOnFile=False)
        start_time = time.time()
        for i in range(0, 2000):
            winner = gameCtrl.playGame()
            if(winner.strategy.name() == "RL-GNN" or winner.strategy.name() == "RL-GNN-STREET" or winner.strategy.name() == "RL-FF"):
                 winner.strategy.epsDecay()
                 print(winner.strategy.getEps())
            idEpisode += 1
            winners.append(winner.id)
            gameCtrl.reset(idEpisode)
            if(i%50==0):
                plotWinners2(winners, strategies) #, rlStrategyFf.name())
            if(i%100==0):
                plt.savefig("plots/wPlot{}.png".format(i)) 
        if(withGraphics):
             pygame.quit()
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")
        # plt.savefig("plots/wPlot.png")

        


