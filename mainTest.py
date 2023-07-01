
from Classes.Strategy.EurPlayer import EuristicPlayer
from Classes.Strategy.RLStrategyFF import ReinforcementLearningStrategyFf
from Classes.Strategy.RLStrategyFFhier import ReinforcementLearningStrategyFfHier
from Classes.Strategy.RLStrategyGNN import ReinforcementLearningStrategyGnn
from Classes.Strategy.RLStrategyGNNhier import ReinforcementLearningStrategyGnnHier
from Classes.Strategy.RLStrategyRGCNhier import ReinforcementLearningStrategyRgcnHier
from Classes.Strategy.RanPlayer import RandomPlayer
import Classes as c

# EURISTIC_PLAYER = EuristicPlayer()
RANDOM_PLAYER = RandomPlayer()

withGraphics = False

# HFF.loadWeights("Weights/HierFF/weights0-4000")
# HFF.loadWeights("Weights/selfHFF_sub/weights0-4000")
# HFF.loadWeights("Weights_without_subgoal/HierFF/weights0-4000")
# HFF.loadWeights("Weights/HierGnn/weights0-4000")

selfHff_sbu = ReinforcementLearningStrategyFfHier(0)
selfHff_sbu.loadWeights("Weights/selfHFF_sub/weights0-4000")

HFF_sub = ReinforcementLearningStrategyFfHier(0)
HFF_sub.loadWeights("Weights/HierFF/weights0-4000")

HFF = ReinforcementLearningStrategyFfHier(0)
HFF.loadWeights("Weights_without_subgoal/HierFF/weights0-4000")

OFF = ReinforcementLearningStrategyFf(0)
OFF.loadWeights("Weights_without_subgoal/OrchFF/weights0-4000")

HGNN = ReinforcementLearningStrategyGnnHier(0)
HGNN.loadWeights("Weights_without_subgoal/HierGnn/weights0-4000")

HGNN_sub = ReinforcementLearningStrategyGnnHier(0)
HGNN_sub.loadWeights("Weights/HierGnn/weights0-4000")

OGNN = ReinforcementLearningStrategyGnn(0)
OGNN.loadWeights("Weights_without_subgoal/OrchGnn/weights0-4000")

HRGCN = ReinforcementLearningStrategyRgcnHier(0)
HRGCN.loadWeights("Weights/HierRGCN/weights0-4000")

HRGCN_sub = ReinforcementLearningStrategyRgcnHier(0)
HRGCN_sub.loadWeights("Weights/HierRGCN_sub/weights0-4000")

examAgent = HFF
against = RandomPlayer()
counter = 0
strategies = [examAgent, against] 
gameCtrl = c.GameController.GameController(playerStrategies = strategies, idEpisode = 0, withGraphics=withGraphics, speed=True)
for i in range(0,5):
    gameCtrl.reset() 
    res = gameCtrl.playTurnamentGame()
    # print(res[0])
    if(res[0] == examAgent.name()):
        counter += 1
strategies = [against, examAgent] 
gameCtrl = c.GameController.GameController(playerStrategies = strategies, idEpisode = 0, withGraphics=withGraphics, speed=True)
for i in range(0,5):
    gameCtrl.reset() 
    res = gameCtrl.playTurnamentGame()
    # print(res[0])
    if(res[0] == examAgent.name()):
        counter += 1
print(counter)
# RESULTS:
# DQN slef FF/FF* = 8-2
# DQN FF/FF* = 9-1
# DQN FF/FF = 7-3
# DQN RGCN/RGCN = 5-5
# DQN GCN/GCN = 8-2
# DQN RGCN/RGCN* = 3-7
# DQN GCN/GCN* = 3-7
# DQN GCN/RAN
# DQN FF/RAN
# RAN/RAN
