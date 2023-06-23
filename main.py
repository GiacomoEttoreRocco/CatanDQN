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
    # HIERARCHICALFF VS RANDOM
        # rlStrategyFfHier = ReinforcementLearningStrategyFfHier()
        # # rEuristic = EuristicPlayer()
        # randomPlayer = RandomPlayer()
        # strategies = [rlStrategyFfHier, randomPlayer] #, rEuristic]
        # withGraphics = False # True    
        # idEpisode = 0
        # gameCtrl = c.GameController.GameController(playerStrategies = strategies, idEpisode = idEpisode, withGraphics=withGraphics, speed=True)
        # for seed in range(1, 11):
        #     winrates = [0,0]
        #     # print("Starting. Eps should be 1: ", .getEps()) # questo 
        #     print("Starting. Eps should be 1: ", rlStrategyFfHier.getEps()) # questo 
        #     saveInCsv([strategies[0].name(), strategies[1].name()], "csvFolder/results"+str(seed)+".csv")
        #     for i in range(0, 300):
        #         print(".", end='', flush = True)
        #         finalPoints = gameCtrl.playGameForTraining()
        #         saveInCsv(finalPoints, "csvFolder/results"+str(seed)+".csv")
        #         if(finalPoints[0] > finalPoints[1]):
        #             winrates[0]+=1
        #         else:
        #             winrates[1]+=1
        #         gameCtrl.reset(strategies)
        #     print("Winrates: ", winrates)
        #     # print("Definitely updated, final eps: ", .getEps(), flush = True)
        #     print("Definitely updated, final eps: ", rlStrategyFfHier.getEps(), flush = True)
        #     # rlStrategyGnn = ReinforcementLearningStrategyGnn()
        #     rlStrategyFfHier = ReinforcementLearningStrategyFfHier()

        #     strategies = [rlStrategyFfHier, randomPlayer] # randomStrategy] #, rEuristic]
        #     gameCtrl.reset(strategies)

###################################################################################################################################################

    # OrchFF VS EURISTIC
        rlStrategyFf = ReinforcementLearningStrategyFf()
        rEuristic = EuristicPlayer()
        # randomPlayer = RandomPlayer()
        strategies = [rlStrategyFf, rEuristic] #, rEuristic]
        withGraphics = False # True    
        idEpisode = 0
        gameCtrl = c.GameController.GameController(playerStrategies = strategies, idEpisode = idEpisode, withGraphics=withGraphics, speed=True)
        for seed in range(1, 11):
            winrates = [0,0]
            # print("Starting. Eps should be 1: ", .getEps()) # questo 
            print("Starting. Eps should be 1: ", rlStrategyFf.getEps()) # questo 
            saveInCsv([strategies[0].name(), strategies[1].name()], "csvFolder/OrchFFVsEur/results"+str(seed)+".csv")
            for i in range(0, 300):
                print(".", end='', flush = True)
                finalPoints = gameCtrl.playGameForTraining()
                saveInCsv(finalPoints, "csvFolder/OrchFFVsEur/results"+str(seed)+".csv")
                if(finalPoints[0] > finalPoints[1]):
                    winrates[0]+=1
                else:
                    winrates[1]+=1
                gameCtrl.reset(strategies)
            print("Winrates: ", winrates)
            # print("Definitely updated, final eps: ", .getEps(), flush = True)
            print("Definitely updated, final eps: ", rlStrategyFf.getEps(), flush = True)
            # rlStrategyGnn = ReinforcementLearningStrategyGnn()
            rlStrategyFf = ReinforcementLearningStrategyFf()

            strategies = [rlStrategyFf, rEuristic] # randomStrategy] #, rEuristic]
            gameCtrl.reset(strategies)

    # HierarchicalGNN VS EURISTIC
        rlGnnHier = RLStrategyGnnHierarchical()
        rEuristic = EuristicPlayer()
        # randomPlayer = RandomPlayer()
        strategies = [rlGnnHier, rEuristic] #, rEuristic]
        withGraphics = False # True    
        idEpisode = 0
        gameCtrl = c.GameController.GameController(playerStrategies = strategies, idEpisode = idEpisode, withGraphics=withGraphics, speed=True)
        for seed in range(1, 11):
            winrates = [0,0]
            # print("Starting. Eps should be 1: ", .getEps()) # questo 
            print("Starting. Eps should be 1: ", rlGnnHier.getEps()) # questo 
            saveInCsv([strategies[0].name(), strategies[1].name()], "csvFolder/GnnHierVsEur/results"+str(seed)+".csv")
            for i in range(0, 300):
                print(".", end='', flush = True)
                finalPoints = gameCtrl.playGameForTraining()
                saveInCsv(finalPoints, "csvFolder/GnnHierVsEur/results"+str(seed)+".csv")
                if(finalPoints[0] > finalPoints[1]):
                    winrates[0]+=1
                else:
                    winrates[1]+=1
                gameCtrl.reset(strategies)
            print("Winrates: ", winrates)
            # print("Definitely updated, final eps: ", .getEps(), flush = True)
            print("Definitely updated, final eps: ", rlGnnHier.getEps(), flush = True)
            # rlStrategyGnn = ReinforcementLearningStrategyGnn()
            rlGnnHier = RLStrategyGnnHierarchical()

            strategies = [rlGnnHier, rEuristic] # randomStrategy] #, rEuristic]
            gameCtrl.reset(strategies)

    # # HIERARCHICALFF VS EURISTIC
        rlStrategyFfHier = ReinforcementLearningStrategyFfHier()
        rEuristic = EuristicPlayer()
        # randomPlayer = RandomPlayer()
        strategies = [rlStrategyFfHier, rEuristic] #, rEuristic]
        withGraphics = False # True    
        idEpisode = 0
        gameCtrl = c.GameController.GameController(playerStrategies = strategies, idEpisode = idEpisode, withGraphics=withGraphics, speed=True)
        for seed in range(1, 11):
            winrates = [0,0]
            # print("Starting. Eps should be 1: ", .getEps()) # questo 
            print("Starting. Eps should be 1: ", rlStrategyFfHier.getEps()) # questo 
            saveInCsv([strategies[0].name(), strategies[1].name()], "csvFolder/HierFFVsEur/results"+str(seed)+".csv")
            for i in range(0, 300):
                print(".", end='', flush = True)
                finalPoints = gameCtrl.playGameForTraining()
                saveInCsv(finalPoints, "csvFolder/HierFFVsEur/results"+str(seed)+".csv")
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

    # # HIERARCHICALGNN VS RANDOM
        rlStrategyGnnHier = RLStrategyGnnHierarchical()
        # rEuristic = EuristicPlayer()
        randomPlayer = RandomPlayer()
        strategies = [rlStrategyGnnHier, randomPlayer] #, rEuristic]
        withGraphics = False # True    
        idEpisode = 0
        gameCtrl = c.GameController.GameController(playerStrategies = strategies, idEpisode = idEpisode, withGraphics=withGraphics, speed=True)
        for seed in range(1, 11):
            winrates = [0,0]
            # print("Starting. Eps should be 1: ", .getEps()) # questo 
            print("Starting. Eps should be 1: ", rlStrategyGnnHier.getEps()) # questo 
            saveInCsv([strategies[0].name(), strategies[1].name()], "csvFolder/HierGnnVsRan/results"+str(seed)+".csv")
            for i in range(0, 300):
                print(".", end='', flush = True)
                finalPoints = gameCtrl.playGameForTraining()
                saveInCsv(finalPoints, "csvFolder/HierGnnVsRan/results"+str(seed)+".csv")
                if(finalPoints[0] > finalPoints[1]):
                    winrates[0]+=1
                else:
                    winrates[1]+=1
                gameCtrl.reset(strategies)
            print("Winrates: ", winrates)
            # print("Definitely updated, final eps: ", .getEps(), flush = True)
            print("Definitely updated, final eps: ", rlStrategyGnnHier.getEps(), flush = True)
            # rlStrategyGnn = ReinforcementLearningStrategyGnn()
            rlStrategyGnnHier = RLStrategyGnnHierarchical()

            strategies = [rlStrategyGnnHier, randomPlayer] # randomStrategy] #, rEuristic]
            gameCtrl.reset(strategies)

    # ORCHESTRATORFF VS RANDOM 
        rlStrategyFf = ReinforcementLearningStrategyFf()

        # rEuristic = EuristicPlayer()
        randomPlayer = RandomPlayer()
        strategies = [rlStrategyFf, randomPlayer] #, rEuristic]
        withGraphics = False # True    
        idEpisode = 0
        gameCtrl = c.GameController.GameController(playerStrategies = strategies, idEpisode = idEpisode, withGraphics=withGraphics, speed=True)
        for seed in range(1, 11):
            winrates = [0,0]
            # print("Starting. Eps should be 1: ", .getEps()) # questo 
            print("Starting. Eps should be 1: ", rlStrategyFf.getEps()) # questo 
            saveInCsv([strategies[0].name(), strategies[1].name()], "csvFolder/OrchFFVsRan/results"+str(seed)+".csv")
            for i in range(0, 300):
                print(".", end='', flush = True)
                finalPoints = gameCtrl.playGameForTraining()
                saveInCsv(finalPoints, "csvFolder/OrchFFVsRan/results"+str(seed)+".csv")
                if(finalPoints[0] > finalPoints[1]):
                    winrates[0]+=1
                else:
                    winrates[1]+=1
                gameCtrl.reset(strategies)
            print("Winrates: ", winrates)
            # print("Definitely updated, final eps: ", .getEps(), flush = True)
            print("Definitely updated, final eps: ", rlStrategyFf.getEps(), flush = True)
            rlStrategyFf = ReinforcementLearningStrategyFf()

            strategies = [rlStrategyFf, randomPlayer] # randomStrategy] #, rEuristic]
            gameCtrl.reset(strategies)

    # # ORCHESTRATORGNN VS RANDOM 
    #     rlStrategyGnn = ReinforcementLearningStrategyGnn()

    #     # rEuristic = EuristicPlayer()
    #     randomPlayer = RandomPlayer()
    #     strategies = [rlStrategyGnn, randomPlayer] #, rEuristic]
    #     withGraphics = False # True    
    #     idEpisode = 0
    #     gameCtrl = c.GameController.GameController(playerStrategies = strategies, idEpisode = idEpisode, withGraphics=withGraphics, speed=True)
    #     for seed in range(1, 11):
    #         winrates = [0,0]
    #         # print("Starting. Eps should be 1: ", .getEps()) # questo 
    #         print("Starting. Eps should be 1: ", rlStrategyGnn.getEps()) # questo 
    #         saveInCsv([strategies[0].name(), strategies[1].name()], "csvFolder/OrchGnnVsRan/results"+str(seed)+".csv")
    #         for i in range(0, 300):
    #             print(".", end='', flush = True)
    #             finalPoints = gameCtrl.playGameForTraining()
    #             saveInCsv(finalPoints, "csvFolder/OrchGnnVsRan/results"+str(seed)+".csv")
    #             if(finalPoints[0] > finalPoints[1]):
    #                 winrates[0]+=1
    #             else:
    #                 winrates[1]+=1
    #             gameCtrl.reset(strategies)
    #         print("Winrates: ", winrates)
    #         # print("Definitely updated, final eps: ", .getEps(), flush = True)
    #         print("Definitely updated, final eps: ", rlStrategyGnn.getEps(), flush = True)
    #         rlStrategyGnn = ReinforcementLearningStrategyGnn()

    #         strategies = [rlStrategyGnn, randomPlayer] # randomStrategy] #, rEuristic]
    #         gameCtrl.reset(strategies)

    # # ORCHESTRATORGNN VS EURISTIC
    #     rlStrategyGnn = ReinforcementLearningStrategyGnn()

    #     rEuristic = EuristicPlayer()
    #     # randomPlayer = RandomPlayer()
    #     strategies = [rlStrategyGnn, rEuristic] #, rEuristic]
    #     withGraphics = False # True    
    #     idEpisode = 0
    #     gameCtrl = c.GameController.GameController(playerStrategies = strategies, idEpisode = idEpisode, withGraphics=withGraphics, speed=True)
    #     for seed in range(1, 11):
    #         winrates = [0,0]
    #         # print("Starting. Eps should be 1: ", .getEps()) # questo 
    #         print("Starting. Eps should be 1: ", rlStrategyGnn.getEps()) # questo 
    #         saveInCsv([strategies[0].name(), strategies[1].name()], "csvFolder/OrchGnnVsEur/results"+str(seed)+".csv")
    #         for i in range(0, 300):
    #             print(".", end='', flush = True)
    #             finalPoints = gameCtrl.playGameForTraining()
    #             saveInCsv(finalPoints, "csvFolder/OrchGnnVsEur/results"+str(seed)+".csv")
    #             if(finalPoints[0] > finalPoints[1]):
    #                 winrates[0]+=1
    #             else:
    #                 winrates[1]+=1
    #             gameCtrl.reset(strategies)
    #         print("Winrates: ", winrates)
    #         # print("Definitely updated, final eps: ", .getEps(), flush = True)
    #         print("Definitely updated, final eps: ", rlStrategyGnn.getEps(), flush = True)
    #         rlStrategyGnn = ReinforcementLearningStrategyGnn()

    #         strategies = [rlStrategyGnn, rEuristic] # randomStrategy] #, rEuristic]
    #         gameCtrl.reset(strategies)


# # RANDOM VS EURISTIC
#         # rlStrategyGnn = ReinforcementLearningStrategyGnn()

#         rEuristic = EuristicPlayer()
#         randomPlayer = RandomPlayer()
#         strategies = [randomPlayer, rEuristic] #, rEuristic]
#         withGraphics = False # True    
#         idEpisode = 0
#         gameCtrl = c.GameController.GameController(playerStrategies = strategies, idEpisode = idEpisode, withGraphics=withGraphics, speed=True)
#         for seed in range(1, 11):
#             winrates = [0,0]
#             # print("Starting. Eps should be 1: ", .getEps()) # questo 
#             # print("Starting. Eps should be 1: ", rlStrategyGnn.getEps()) # questo 
#             saveInCsv([strategies[0].name(), strategies[1].name()], "csvFolder/RanVsEur/results"+str(seed)+".csv")
#             for i in range(0, 300):
#                 print(".", end='', flush = True)
#                 finalPoints = gameCtrl.playGameForTraining()
#                 saveInCsv(finalPoints, "csvFolder/RanVsEur/results"+str(seed)+".csv")
#                 if(finalPoints[0] > finalPoints[1]):
#                     winrates[0]+=1
#                 else:
#                     winrates[1]+=1
#                 gameCtrl.reset(strategies)
#             print("Winrates: ", winrates)
#             # print("Definitely updated, final eps: ", .getEps(), flush = True)
#             # print("Definitely updated, final eps: ", rlStrategyGnn.getEps(), flush = True)
#             # rlStrategyGnn = ReinforcementLearningStrategyGnn()

#             strategies = [randomPlayer, rEuristic] # randomStrategy] #, rEuristic]
#             gameCtrl.reset(strategies)


# # EURISTIC VS RANDOM 
        # # rlStrategyGnn = ReinforcementLearningStrategyGnn()

        # rEuristic = EuristicPlayer()
        # randomPlayer = RandomPlayer()
        # strategies = [rEuristic, randomPlayer] #, rEuristic]
        # withGraphics = False # True    
        # idEpisode = 0
        # gameCtrl = c.GameController.GameController(playerStrategies = strategies, idEpisode = idEpisode, withGraphics=withGraphics, speed=True)
        # for seed in range(1, 11):
        #     winrates = [0,0]
        #     # print("Starting. Eps should be 1: ", .getEps()) # questo 
        #     # print("Starting. Eps should be 1: ", rlStrategyGnn.getEps()) # questo 
        #     saveInCsv([strategies[0].name(), strategies[1].name()], "csvFolder/EurVsRan/results"+str(seed)+".csv")
        #     for i in range(0, 300):
        #         print(".", end='', flush = True)
        #         finalPoints = gameCtrl.playGameForTraining()
        #         saveInCsv(finalPoints, "csvFolder/EurVsRan/results"+str(seed)+".csv")
        #         if(finalPoints[0] > finalPoints[1]):
        #             winrates[0]+=1
        #         else:
        #             winrates[1]+=1
        #         gameCtrl.reset(strategies)
        #     print("Winrates: ", winrates)
        #     # print("Definitely updated, final eps: ", .getEps(), flush = True)
        #     # print("Definitely updated, final eps: ", rlStrategyGnn.getEps(), flush = True)
        #     # rlStrategyGnn = ReinforcementLearningStrategyGnn()

        #     strategies = [rEuristic, randomPlayer] # randomStrategy] #, rEuristic]
        #     gameCtrl.reset(strategies)
        