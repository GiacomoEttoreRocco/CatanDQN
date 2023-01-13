import Classes as c
from AI.Gnn import Gnn

import pandas as pd

PURE = True
toggle = False
toVis = False

if(not toVis):
    speed = True
    withGraphics = False
    withDelay = False
    realPlayer = False
    save = True
    train = True
else:
    speed =  True # False #
    withGraphics = True
    withDelay = False
    realPlayer = False
    save = False
    train = False

WINNERS = [0.0, 0.0, 0.0, 0.0]

def printWinners():
        normValue = sum(WINNERS)
        toPrint = [0.0, 0.0, 0.0, 0.0]
        for i, v in enumerate(WINNERS):
            toPrint[i] = v/normValue
        s = ""
        for i, vperc in enumerate(toPrint):
            s = s + "Player " + str(i+1)+ ": " + str(round(vperc*100,2)) + " % "
        print(s)
        print(WINNERS)

iterations = 50
numberTrainGame = 1
numberTestGame = 1

if __name__ == '__main__':
    for epoch in range(iterations):
        print('Iteration: ', epoch+1, "/", iterations)
        allGames = pd.DataFrame(data={'places': [], 'edges':[], 'globals':[]})   
        for batch in range(numberTrainGame): 
            print('game: ', batch+1, "/", numberTrainGame) 
            game = c.GameController.GameController(withGraphics=True)
            winnerId = game.playGameWithGraphic()
            WINNERS[winnerId-1]+=1
            allGames = pd.concat([allGames, game.total], ignore_index=True)
            print("Length of total moves of allGames: ", len(allGames))

        if(train):
            printWinners()
            allGames.to_json("./json/training_game.json")

        allGames = pd.DataFrame(data={'places': [], 'edges':[], 'globals':[]})   
        for batch in range(numberTestGame): 
            print('game: ', batch+1, "/", numberTestGame) 
            game = c.GameController.GameController(withGraphics=True)
            winnerId = game.playGameWithGraphic()
            WINNERS[winnerId-1]+=1
            allGames = pd.concat([allGames, game.total], ignore_index=True)
            print("Length of total moves of allGames: ", len(allGames))

        if(train):
            printWinners()
            allGames.to_json("./json/testing_game.json")
            Gnn().trainModel(validate=True)
                