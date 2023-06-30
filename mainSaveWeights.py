from matplotlib import pyplot as plt
import pygame
import Classes as c
import pandas as pd
import numpy as np
import csv
from Classes.Strategy.RLStrategyFFhier import ReinforcementLearningStrategyFfHier
from Classes.Strategy.RLStrategyFFhier_mod import ReinforcementLearningStrategyFfHier_mod
from Classes.Strategy.RLStrategyGNN import ReinforcementLearningStrategyGnn
from Classes.Strategy.RLStrategyFF import ReinforcementLearningStrategyFf
import time
from Classes.Strategy.RLStrategyGNNhier import ReinforcementLearningStrategyGnnHier
from Classes.Strategy.EurPlayer import EuristicPlayer
from Classes.Strategy.RLStrategyGNNhier_mod import ReinforcementLearningStrategyGnnHier_mod
from Classes.Strategy.RLStrategyRGCN import ReinforcementLearningStrategyRgcn
from Classes.Strategy.RLStrategyRGCNhier import ReinforcementLearningStrategyRgcnHier
from Classes.Strategy.RanPlayer import RandomPlayer
from Classes.Strategy.StrategyRandom import StrategyRandom
from Classes.staticUtilities import plotCsvColumns, plotCsvColumnsWithHeaders, plotWinners2, saveInCsv

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
        print("\nStarting. Eps should be 1: ", agent1.getEps(), "\n") 
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
            if(i%100 == 0 and i!=0):
                print("\nEps until now: ", agent1.getEps(), "\n")
            if(i%1000 == 0 and i!=0):
                agent1.saveWeights("Weights/"+nameOfTheFolder+"/weights"+str(seed)+"-"+str(i))

        # print("Winrates: ", winrates)
        print("\nDefinitely updated, final eps: ", agent1.getEps(), flush = True)
        agent1.saveWeights("Weights/"+nameOfTheFolder+"/weights"+str(seed)+"-4000")
        agent1.__init__(1)
        # rlStrategyFfHier = ReinforcementLearningStrategyFfHier(1)
        strategies = [agent1, agent2] 
        gameCtrl.resetAndResetStrategies(strategies)


def randomAndEuristic(outFor, inFor, agent1, agent2, nameOfTheFolder):
    strategies = [agent1, agent2] #, rEuristic]
    withGraphics = False # True    
    idEpisode = 0
    gameCtrl = c.GameController.GameController(playerStrategies = strategies, idEpisode = idEpisode, withGraphics=withGraphics, speed=True)
    for seed in range(1, 5):
        winrates = [0,0]
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
        # rlStrategyFfHier = ReinforcementLearningStrategyFfHier(1)
        strategies = [agent1, agent2] 
        gameCtrl.resetAndResetStrategies(strategies)

if __name__ == '__main__':
        
    outFor = 3 # SETTA ANCHE L'INIZIO, FACCIAMO SOLO CONTRO I RANDOM
    inFor = 1000

    trainAndSaveWeights(outFor, inFor, ReinforcementLearningStrategyRgcnHier(1), RandomPlayer(), "HierRGCN")
    
    # trainAndSaveWeights(outFor, inFor, ReinforcementLearningStrategyGnnHier(1), RandomPlayer(), "HierGnn")
    # trainAndSaveWeights(outFor, inFor, ReinforcementLearningStrategyFfHier(1), RandomPlayer(), "HierFF")

    # trainAndSaveWeights(outFor, inFor, ReinforcementLearningStrategyGnnHier_mod(1), RandomPlayer(), "HierGnn_mod")
    # trainAndSaveWeights(outFor, inFor, ReinforcementLearningStrategyFfHier_mod(1), RandomPlayer(), "HierFF_mod")

    # trainAndSaveWeights(outFor, inFor, ReinforcementLearningStrategyGnn(1), RandomPlayer(), "OrchGnn")
    # trainAndSaveWeights(outFor, inFor, ReinforcementLearningStrategyFf(1), RandomPlayer(), "OrchFF")

    # randomAndEuristic(outFor, inFor, RandomPlayer(), RandomPlayer(), "Ran")


    # trainAndSaveWeights(outFor, inFor, ReinforcementLearningStrategyFfHier(1), RandomPlayer(), "longTrainFF")
    # trainAndSaveWeights(outFor, inFor, ReinforcementLearningStrategyGnnHier(1), RandomPlayer(), "longTrainGnn")

