
from Classes.Strategy.EurPlayer import EuristicPlayer
from Classes.Strategy.RLStrategyFF import ReinforcementLearningStrategyFf
from Classes.Strategy.RLStrategyFFhier import ReinforcementLearningStrategyFfHier
from Classes.Strategy.RLStrategyGNN import ReinforcementLearningStrategyGnn
from Classes.Strategy.RLStrategyGNNhier import ReinforcementLearningStrategyGnnHier
from Classes.Strategy.RanPlayer import RandomPlayer
import Classes as c

# def append_to_text_file(file_path, text, data_list):
#     with open(file_path, 'a') as file:
#         file.write(text + ': ')
#         for item in data_list:
#             file.write(str(item) + ' ')
#         file.write("\n")

# # ReinforcementLearningStrategyFf
# # ReinforcementLearningStrategyFfHier
# # ReinforcementLearningStrategyGnn
# # ReinforcementLearningStrategyGnnHier
# # EuristicPlayer
# # RandomPlayer

# H_FF1_blue = ReinforcementLearningStrategyFfHier(0)
# H_FF1_blue.loadWeights("Weights/HierFFVsEur/weights"+str(1))
# H_FF2_blue = ReinforcementLearningStrategyFfHier(0)
# H_FF2_blue.loadWeights("Weights/HierFFVsEur/weights"+str(2))

# H_GNN1_blue = ReinforcementLearningStrategyGnnHier(0)
# H_GNN1_blue.loadWeights("Weights/HierGnnVsEur/weights"+str(1))
# H_GNN2_blue = ReinforcementLearningStrategyGnnHier(0)
# H_GNN2_blue.loadWeights("Weights/HierGnnVsEur/weights"+str(2))

# O_FF1_blue = ReinforcementLearningStrategyFf(0)
# O_FF1_blue.loadWeights("Weights/OrchFFVsEur/weights"+str(1))
# O_FF2_blue = ReinforcementLearningStrategyFf(0)
# O_FF2_blue.loadWeights("Weights/OrchFFVsEur/weights"+str(2))

O_GNN1_blue = ReinforcementLearningStrategyGnn(0)
O_GNN1_blue.loadWeights("Weights/OrchGnnVsEur/weights"+str(1))
# O_GNN2_blue = ReinforcementLearningStrategyGnn(0)
# O_GNN2_blue.loadWeights("Weights/OrchGnnVsEur/weights"+str(2))

# H_FF1_green = ReinforcementLearningStrategyFfHier(0)
# H_FF1_green.loadWeights("Weights/HierFFVsRan/weights"+str(1))
# H_FF2_green = ReinforcementLearningStrategyFfHier(0)
# H_FF1_green.loadWeights("Weights/HierFFVsRan/weights"+str(2))

# H_GNN1_green = ReinforcementLearningStrategyGnnHier(0)
# H_GNN1_green.loadWeights("Weights/HierGnnVsRan/weights"+str(1))
# H_GNN2_green = ReinforcementLearningStrategyGnnHier(0)
# H_GNN2_green.loadWeights("Weights/HierGnnVsRan/weights"+str(2))

# O_FF1_green = ReinforcementLearningStrategyFf(0)
# O_FF1_green.loadWeights("Weights/OrchFFVsRan/weights"+str(1))
# O_FF2_green = ReinforcementLearningStrategyFf(0)
# O_FF2_green.loadWeights("Weights/OrchFFVsRan/weights"+str(2))

# O_GNN1_green = ReinforcementLearningStrategyGnn(0)
# O_GNN1_green.loadWeights("Weights/OrchGnnVsRan/weights"+str(1))
# O_GNN2_green = ReinforcementLearningStrategyGnn(0)
# O_GNN2_green.loadWeights("Weights/OrchGnnVsRan/weights"+str(2))

EURISTIC_PLAYER = EuristicPlayer()
RANDOM_PLAYER = RandomPlayer()

# H_GNN1_self = ReinforcementLearningStrategyGnnHier(0)
# H_GNN1_self.loadWeights("Weights/HierGnnVsHierGnn/weights"+str(1))

# H_GNN2_self = ReinforcementLearningStrategyGnnHier(0)
# H_GNN2_self.loadWeights("Weights/HierGnnVsHierGnn/weights"+str(2))

# blueAgents = [H_FF1_blue, H_FF2_blue, H_GNN1_blue, H_GNN2_blue, O_FF1_blue, O_FF2_blue, O_GNN1_blue, O_GNN2_blue]
# greenAgents = [H_FF1_green, H_FF2_green, H_GNN1_green, H_GNN2_green, O_FF1_green, O_FF2_green, O_GNN1_green, O_GNN2_green]

# allAgents = [H_FF1_blue, H_FF2_blue, H_GNN1_blue, H_GNN2_blue, O_FF1_blue, O_FF2_blue, O_GNN1_blue, O_GNN2_blue, H_FF1_green, H_FF2_green, H_GNN1_green, H_GNN2_green, O_FF1_green, O_FF2_green, O_GNN1_green, O_GNN2_green]

#############################################################

withGraphics = True

# otherAgents = [EURISTIC_PLAYER, RANDOM_PLAYER, H_FF2_blue, H_GNN1_blue, H_GNN2_blue, O_FF1_blue, O_FF2_blue, O_GNN1_blue, O_GNN2_blue, H_FF1_green, H_FF2_green, H_GNN1_green, H_GNN2_green, O_FF1_green, O_FF2_green, O_GNN1_green, O_GNN2_green]
examAgent = RandomPlayer()
against = EURISTIC_PLAYER
strategies = [examAgent, against] 
gameCtrl = c.GameController.GameController(playerStrategies = strategies, idEpisode = 0, withGraphics=withGraphics, speed=False)
gameCtrl.reset() 
res = gameCtrl.playTurnamentGame()

