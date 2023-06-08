import Classes as c
from AI.Gnn import Gnn
# from Classes.PlayerTypes import PlayerTypes
import pandas as pd
import numpy as np
import csv
import shutil
from Classes.Strategy.HybridStrategy import HybridStrategy

from Classes.Strategy.PriorityStrategy import PriorityStrategy
from Classes.Strategy.PureStrategy import PureStrategy

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

def training(iterationProcessIndex, iterations, numberOfTrainingGames, numberOfValidationGames):
    winners = [0.0,0.0,0.0,0.0]
    playerStrategies = [PriorityStrategy, PriorityStrategy, HybridStrategy, PureStrategy]
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

# def simulationMain():
#     SHOW = True

#     if not SHOW:
#         numberOfRepetitions = 1
#         maxPerformanceResults = 0

#         for idx in range(numberOfRepetitions):
#             training(idx, iterations=2, numberOfTrainingGames=5, numberOfValidationGames=5)
            
#             results = []
#             playerTypes = [PlayerTypes.PRIORITY, PlayerTypes.PRIORITY, PlayerTypes.PRIORITY, PlayerTypes.HYBRID]
#             results.append(performanceEvaluation(idx, playerTypes=playerTypes, numberOfTestingGames=15, withGraphics=False))

#             playerTypes = [PlayerTypes.PRIORITY, PlayerTypes.PRIORITY, PlayerTypes.PRIORITY, PlayerTypes.PURE]
#             results.append(performanceEvaluation(idx, playerTypes=playerTypes, numberOfTestingGames=15, withGraphics=False))
            
#             playerTypes = [PlayerTypes.HYBRID, PlayerTypes.HYBRID, PlayerTypes.HYBRID, PlayerTypes.PURE]
#             results.append(performanceEvaluation(idx, playerTypes=playerTypes, numberOfTestingGames=15, withGraphics=False))

#             if maxPerformanceResults<sum(results):
#                 print(f'Saving best weights in iteration {idx}...')
#                 maxPerformanceResults = sum(results)
#                 shutil.copyfile('AI/model_weights.pth', 'AI/best_model_weights.pth')

#             writeOnCsv(idx, results)
#     else:
#         for idx in range(1):
#             results = []
#             playerTypes = [PlayerTypes.PRIORITY, PlayerTypes.PRIORITY, PlayerTypes.PRIORITY, PlayerTypes.HYBRID]
#             results.append(performanceEvaluation(0, playerTypes=playerTypes, numberOfTestingGames=25, withGraphics=True))

#             playerTypes = [PlayerTypes.PRIORITY, PlayerTypes.PRIORITY, PlayerTypes.PRIORITY, PlayerTypes.PURE]
#             results.append(performanceEvaluation(0, playerTypes=playerTypes, numberOfTestingGames=25, withGraphics=True))
            
#             playerTypes = [PlayerTypes.HYBRID, PlayerTypes.HYBRID, PlayerTypes.HYBRID, PlayerTypes.PURE]
#             results.append(performanceEvaluation(0, playerTypes=playerTypes, numberOfTestingGames=25, withGraphics=True))

        #     with open('best_results.csv', 'a') as f:
        #         writer = csv.writer(f)
        #         writer.writerow(results)

        # Gnn().modelWeightsPath = "AI/best_model_weights.pth"
        # playerTypes = [PlayerTypes.PRIORITY,PlayerTypes.PRIORITY,PlayerTypes.PRIORITY,PlayerTypes.HYBRID]
        # performanceEvaluation(0, playerTypes=playerTypes, numberOfTestingGames=1, withGraphics=True, speed=True)

if __name__ == '__main__':
        # types = [PlayerTypes.HYBRID, PlayerTypes.HYBRID, PlayerTypes.HYBRID, PlayerTypes.HYBRID]
        # types = [PlayerTypes.HYBRID, PlayerTypes.HYBRID] #, PlayerTypes.HYBRID, PlayerTypes.HYBRID]
        # types = [PlayerTypes.PRIORITY, PlayerTypes.PURE]
        prioStrategy = PriorityStrategy()
        hybStrategy = HybridStrategy()
        purStrategy = PureStrategy()
        strategies = [hybStrategy, purStrategy]
        gameCtrl = c.GameController.GameController(playerStrategies = strategies, withGraphics=True, speed=True, saveOnFile=False)
        
        winner = gameCtrl.playGame()
