
from Classes.Strategy.RLStrategyFFhier import ReinforcementLearningStrategyFfHier
from Classes.Strategy.RanPlayer import RandomPlayer
import Classes as c

withGraphics = True

def append_to_text_file(file_path, text, data_list):
    with open(file_path, 'a') as file:
        file.write(text + '\n')
        for item in data_list:
            file.write(str(item) + '\n')

def doGame(agent1, agent2, path1, path2):
    strategies = [agent1, agent2]

    gameCtrl = c.GameController.GameController(playerStrategies = strategies, idEpisode = 0, withGraphics=withGraphics, speed=True)
    if("RL" in agent1.name()):
        agent1 = agent1.loadWeights(path1)
    if("RL" in agent2.name()):
        agent2 = agent2.loadWeights(path2)
    return gameCtrl.playTurnamentGame()

rlStrategyFfHier = ReinforcementLearningStrategyFfHier()
randomPlayer = RandomPlayer()

rlStrategyFfHier = ReinforcementLearningStrategyFfHier()
randomPlayer = RandomPlayer()

for i in range(0, 5):
    res = doGame(rlStrategyFfHier, randomPlayer, "Weights/HierFFVsRan/weights"+str(1), "")
    # append_to_text_file("Torneo.txt", "HierFFVsRandom", res)
    rlStrategyFfHier = ReinforcementLearningStrategyFfHier()
    randomPlayer = RandomPlayer()

print(res)