
from Classes.Strategy.EurPlayer import EuristicPlayer
from Classes.Strategy.RLStrategyFF import ReinforcementLearningStrategyFf
from Classes.Strategy.RLStrategyFFhier import ReinforcementLearningStrategyFfHier
from Classes.Strategy.RLStrategyGNN import ReinforcementLearningStrategyGnn
from Classes.Strategy.RLStrategyGNNhier import ReinforcementLearningStrategyGnnHier
from Classes.Strategy.RLStrategyRGCNhier import ReinforcementLearningStrategyRgcnHier
from Classes.Strategy.RanPlayer import RandomPlayer
import Classes as c

withGraphics = False

selfHff_sub = ReinforcementLearningStrategyFfHier(0)
selfHff_sub.loadWeights("Weights/selfHFF_sub/weights2-4000")

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

randomPlayer = RandomPlayer()

allAgents = [selfHff_sub, HFF_sub, HFF, OFF, HGNN_sub, HGNN, OGNN, HRGCN, HRGCN_sub, randomPlayer]

# a1 = selfHff_sub
a1 = HFF_sub

for a2 in allAgents:
    strategies = [a1, a2] 
    counterA1 = 0
    counterA2 = 0

    gameCtrl = c.GameController.GameController(playerStrategies = strategies, idEpisode = 0, withGraphics=withGraphics, speed=True)
    for i in range(0,5):
        gameCtrl.reset() 
        res = gameCtrl.playTurnamentGame()
        # print(res[0])
        if(res[0] == a1.name()):
            counterA1 += 1
        else: counterA2 += 1

    print(a1.name(), "-", a2.name(), ": ", counterA1, " ", counterA2)




