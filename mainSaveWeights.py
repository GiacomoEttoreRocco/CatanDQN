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
from Classes.Strategy.RLStrategyGNNhier import ReinforcementLearningStrategyGnnHier
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

def trainAndSaveWeights(outFor, inFor, agent1, agent2, nameOfTheFolder):
    strategies = [agent1, agent2] #, rEuristic]
    withGraphics = False # True    
    idEpisode = 0
    gameCtrl = c.GameController.GameController(playerStrategies = strategies, idEpisode = idEpisode, withGraphics=withGraphics, speed=True)
    for seed in range(1, outFor):
        winrates = [0,0]
        print("Starting. Eps should be 1: ", agent1.getEps()) 
        saveInCsv([strategies[0].name(), strategies[1].name()], "csvFolder/"+nameOfTheFolder+"/results"+str(seed)+".csv")
        for i in range(0, inFor):
            finalPoints = gameCtrl.playGameForTraining()
            print(finalPoints[0], end=' ', flush = True)
            saveInCsv(finalPoints, "csvFolder/"+nameOfTheFolder+"/results"+str(seed)+".csv")
            if(finalPoints[0] > finalPoints[1]):
                winrates[0]+=1
            else:
                winrates[1]+=1
            gameCtrl.reset()
        # print("Winrates: ", winrates)
        print("Definitely updated, final eps: ", agent1.getEps(), flush = True)
        agent1.saveWeights("Weights/"+nameOfTheFolder+"/weights"+str(seed))
        agent1.__init__(1)
        # rlStrategyFfHier = ReinforcementLearningStrategyFfHier(1)
        strategies = [agent1, agent2] 
        gameCtrl.resetAndResetStrategies(strategies)

if __name__ == '__main__':
        
    outFor = 5
    inFor = 500

    trainAndSaveWeights(outFor, inFor, ReinforcementLearningStrategyFfHier(1), RandomPlayer(), "HierFFVsRan")
    trainAndSaveWeights(outFor, inFor, ReinforcementLearningStrategyFfHier(1), EuristicPlayer(), "HierFFVsEur")

    trainAndSaveWeights(outFor, inFor, ReinforcementLearningStrategyFf(1), RandomPlayer(), "OrchFFVsRan")
    trainAndSaveWeights(outFor, inFor, ReinforcementLearningStrategyFf(1), EuristicPlayer(), "OrchFFVsEur")

    trainAndSaveWeights(outFor, inFor, ReinforcementLearningStrategyGnnHier(1), RandomPlayer(), "HierGnnVsRan")
    trainAndSaveWeights(outFor, inFor, ReinforcementLearningStrategyGnnHier(1), EuristicPlayer(), "HierGnnVsEur")

    trainAndSaveWeights(outFor, inFor, ReinforcementLearningStrategyGnn(1), RandomPlayer(), "OrchGnnVsRan")
    trainAndSaveWeights(outFor, inFor, ReinforcementLearningStrategyGnn(1), EuristicPlayer(), "OrchGnnVsEur")

    trainAndSaveWeights(outFor, inFor, RandomPlayer(), RandomPlayer(), "RanVsRan")
    trainAndSaveWeights(outFor, inFor, EuristicPlayer(), RandomPlayer(), "EurVsRan")

    trainAndSaveWeights(outFor, inFor, EuristicPlayer(), EuristicPlayer(), "EurVsEur")
    trainAndSaveWeights(outFor, inFor, RandomPlayer(), EuristicPlayer(), "RanVsEur")

