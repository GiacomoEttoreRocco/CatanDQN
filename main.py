import Classes as c
from AI.Gnn import Gnn
from Classes.PlayerTypes import PlayerTypes
import pandas as pd
import numpy as np
import csv

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
    playerTypes = [PlayerTypes.PRIORITY, PlayerTypes.PRIORITY, PlayerTypes.HYBRID, PlayerTypes.PURE]
    print(f'Starting training: {iterationProcessIndex}')
    for iteration in range(iterations):
        print('Iteration: ', iteration+1, "/", iterations)
        allGames = pd.DataFrame(data={'places': [], 'edges':[], 'globals':[]})
        np.random.shuffle(playerTypes)
        for numGame in range(numberOfTrainingGames): 
            print('game: ', numGame+1, "/", numberOfTrainingGames) 
            game = c.GameController.GameController(playerTypes=playerTypes)
            winner = game.playGameWithGraphic()
            winners[winner.id-1]+=1
            allGames = pd.concat([allGames, game.total], ignore_index=True)
        
        print("Length of total moves of allGames: ", len(allGames))
        # printWinners(winners)
        allGames.to_json("./json/training_game.json")

        allGames = pd.DataFrame(data={'places': [], 'edges':[], 'globals':[]})   
        for numGame in range(numberOfValidationGames): 
            print('game: ', numGame+1, "/", numberOfValidationGames) 
            game = c.GameController.GameController(playerTypes=playerTypes)
            winner = game.playGameWithGraphic()
            winners[winner.id-1]+=1
            allGames = pd.concat([allGames, game.total], ignore_index=True)

        print("Length of total moves of allGames: ", len(allGames))
        # printWinners(winners)
        allGames.to_json("./json/testing_game.json")
        
        Gnn().trainModel(validate=True)


def performanceEvaluation(iterationProcessIndex, playerTypes, numberOfTestingGames, withGraphics=True, speed=True):
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
    return [*winners, winners[0]/sum(winners)*100.0]

    
def writeOnCsv(i, winners):
    with open('results.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([i, *winners])

    
if __name__ == '__main__':
    
    # numberOfRepetitions = 30

    # for idx in range(numberOfRepetitions):
    #     training(idx, iterations=2, numberOfTrainingGames=10, numberOfValidationGames=10)
        
    #     results = []
    #     playerTypes = [PlayerTypes.PRIORITY, PlayerTypes.PRIORITY, PlayerTypes.PRIORITY, PlayerTypes.HYBRID]
    #     results.extend(performanceEvaluation(idx, playerTypes=playerTypes, numberOfTestingGames=20, withGraphics=False))

    #     playerTypes = [PlayerTypes.PRIORITY, PlayerTypes.PRIORITY, PlayerTypes.PRIORITY, PlayerTypes.PURE]
    #     results.extend(performanceEvaluation(idx, playerTypes=playerTypes, numberOfTestingGames=20, withGraphics=False))
        
    #     playerTypes = [PlayerTypes.HYBRID, PlayerTypes.HYBRID, PlayerTypes.HYBRID, PlayerTypes.PURE]
    #     results.extend(performanceEvaluation(idx, playerTypes=playerTypes, numberOfTestingGames=20, withGraphics=False))

    #     writeOnCsv(idx, results)

    playerTypes = [PlayerTypes.PURE, PlayerTypes.PURE] #, PlayerTypes.PRIORITY, PlayerTypes.HYBRID]
    performanceEvaluation(0, playerTypes=playerTypes, numberOfTestingGames=20, withGraphics=True)



                