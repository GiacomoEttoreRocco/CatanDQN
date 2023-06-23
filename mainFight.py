
from Classes.Strategy.EurPlayer import EuristicPlayer
from Classes.Strategy.RLStrategyFF import ReinforcementLearningStrategyFf
from Classes.Strategy.RLStrategyFFhier import ReinforcementLearningStrategyFfHier
from Classes.Strategy.RLStrategyGNN import ReinforcementLearningStrategyGnn
from Classes.Strategy.RLStrategyGNNhier import ReinforcementLearningStrategyGnnHier
from Classes.Strategy.RanPlayer import RandomPlayer
import Classes as c

withGraphics = True

def append_to_text_file(file_path, text, data_list):
    with open(file_path, 'a') as file:
        file.write(text + ': ')
        for item in data_list:
            file.write(str(item) + ' ')
        file.write("\n")

def doGame(gameCtrl, agent1, agent2, path1, path2):
    strategies = [agent1, agent2]

    if("RL" in agent1.name()):
        agent1 = agent1.loadWeights(path1)
    if("RL" in agent2.name()):
        agent2 = agent2.loadWeights(path2)
    return gameCtrl.playTurnamentGame()


# ReinforcementLearningStrategyFf
# ReinforcementLearningStrategyFfHier
# ReinforcementLearningStrategyGnn
# ReinforcementLearningStrategyGnnHier
# EuristicPlayer
# RandomPlayer

H_FF1_blue = ReinforcementLearningStrategyFfHier(0)
H_FF1_blue.loadWeights("Weights/HierFFVsEur/weights"+str(1))
H_FF2_blue = ReinforcementLearningStrategyFfHier(0)
H_FF2_blue.loadWeights("Weights/HierFFVsEur/weights"+str(2))
H_FF3_blue = ReinforcementLearningStrategyFfHier(0)
H_FF3_blue.loadWeights("Weights/HierFFVsEur/weights"+str(3))

H_GNN1_blue = ReinforcementLearningStrategyGnnHier(0)
H_GNN1_blue.loadWeights("Weights/HierGnnVsEur/weights"+str(1))
H_GNN2_blue = ReinforcementLearningStrategyGnnHier(0)
H_GNN2_blue.loadWeights("Weights/HierGnnVsEur/weights"+str(2))
H_GNN3_blue = ReinforcementLearningStrategyGnnHier(0)
H_GNN3_blue.loadWeights("Weights/HierGnnVsEur/weights"+str(3))

O_FF1_blue = ReinforcementLearningStrategyFf(0)
O_FF1_blue.loadWeights("Weights/OrchFFVsEur/weights"+str(1))
O_FF2_blue = ReinforcementLearningStrategyFf(0)
O_FF2_blue.loadWeights("Weights/OrchFFVsEur/weights"+str(2))
O_FF3_blue = ReinforcementLearningStrategyFf(0)
O_FF3_blue.loadWeights("Weights/OrchFFVsEur/weights"+str(3))

O_GNN1_blue = ReinforcementLearningStrategyGnn(0)
O_GNN1_blue.loadWeights("Weights/OrchGnnVsEur/weights"+str(1))
O_GNN2_blue = ReinforcementLearningStrategyGnn(0)
O_GNN2_blue.loadWeights("Weights/OrchGnnVsEur/weights"+str(2))
O_GNN3_blue = ReinforcementLearningStrategyGnn(0)
O_GNN3_blue.loadWeights("Weights/OrchGnnVsEur/weights"+str(3))

H_FF1_green = ReinforcementLearningStrategyFfHier(0)
H_FF1_green.loadWeights("Weights/HierFFVsRan/weights"+str(1))
H_FF2_green = ReinforcementLearningStrategyFfHier(0)
H_FF1_green.loadWeights("Weights/HierFFVsRan/weights"+str(2))
H_FF3_green = ReinforcementLearningStrategyFfHier(0)
H_FF3_green.loadWeights("Weights/HierFFVsRan/weights"+str(3))

H_GNN1_green = ReinforcementLearningStrategyGnnHier(0)
H_GNN1_green.loadWeights("Weights/HierGnnVsRan/weights"+str(1))
H_GNN2_green = ReinforcementLearningStrategyGnnHier(0)
H_GNN2_green.loadWeights("Weights/HierGnnVsRan/weights"+str(2))
H_GNN3_green = ReinforcementLearningStrategyGnnHier(0)
H_GNN3_green.loadWeights("Weights/HierGnnVsRan/weights"+str(3))

O_FF1_green = ReinforcementLearningStrategyFf(0)
O_FF1_green.loadWeights("Weights/OrchFFVsRan/weights"+str(1))
O_FF2_green = ReinforcementLearningStrategyFf(0)
O_FF2_green.loadWeights("Weights/OrchFFVsRan/weights"+str(2))
O_FF3_green = ReinforcementLearningStrategyFf(0)
O_FF3_green.loadWeights("Weights/OrchFFVsRan/weights"+str(3))

O_GNN1_green = ReinforcementLearningStrategyGnn(0)
O_GNN1_green.loadWeights("Weights/OrchGnnVsRan/weights"+str(1))
O_GNN2_green = ReinforcementLearningStrategyGnn(0)
O_GNN2_green.loadWeights("Weights/OrchGnnVsRan/weights"+str(2))
O_GNN3_green = ReinforcementLearningStrategyGnn(0)
O_GNN3_green.loadWeights("Weights/OrchGnnVsRan/weights"+str(3))

EURISTIC_PLAYER = EuristicPlayer()
RANDOM_PLAYER = RandomPlayer()




allAgents = []

rlStrategyFfHier = ReinforcementLearningStrategyFfHier(0)
randomPlayer = RandomPlayer()

#Andata
strategies = [rlStrategyFfHier, randomPlayer] 
gameCtrl = c.GameController.GameController(playerStrategies = strategies, idEpisode = 0, withGraphics=withGraphics, speed=True)

for i in range(1, 10):
    res = doGame(gameCtrl, rlStrategyFfHier, randomPlayer, "Weights/HierFFVsRan/weights"+str(1), "")
    append_to_text_file("Torneo.txt", "HierFFVsRandom1", res)
    rlStrategyFfHier = ReinforcementLearningStrategyFfHier(0)
    randomPlayer = RandomPlayer()
    strategies = [rlStrategyFfHier, randomPlayer] 
    gameCtrl.reset(strategies)
    print(res)

#Ritorno

strategies = [randomPlayer, rlStrategyFfHier] 
gameCtrl = c.GameController.GameController(playerStrategies = strategies, idEpisode = 0, withGraphics=withGraphics, speed=True)

for i in range(1, 10):
    # res = doGame(gameCtrl, rlStrategyFfHier, randomPlayer, "Weights/HierFFVsRan/weights"+str(1), "")
    res = doGame(gameCtrl, randomPlayer, rlStrategyFfHier, "", "Weights/HierFFVsRan/weights"+str(1))

    append_to_text_file("Torneo.txt", "HierFFVsRandom1", res)
    rlStrategyFfHier = ReinforcementLearningStrategyFfHier(0)
    randomPlayer = RandomPlayer()
    strategies = [randomPlayer, rlStrategyFfHier] 
    gameCtrl.reset(strategies)
    print(res)