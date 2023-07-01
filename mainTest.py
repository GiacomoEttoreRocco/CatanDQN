
from Classes.Strategy.EurPlayer import EuristicPlayer
from Classes.Strategy.RLStrategyFF import ReinforcementLearningStrategyFf
from Classes.Strategy.RLStrategyFFhier import ReinforcementLearningStrategyFfHier
from Classes.Strategy.RLStrategyGNN import ReinforcementLearningStrategyGnn
from Classes.Strategy.RLStrategyGNNhier import ReinforcementLearningStrategyGnnHier
from Classes.Strategy.RanPlayer import RandomPlayer
import Classes as c

EURISTIC_PLAYER = EuristicPlayer()
RANDOM_PLAYER = RandomPlayer()

withGraphics = False

HFF = ReinforcementLearningStrategyFfHier(0)
# HFF.loadWeights("Weights/HierFF/weights0-4000")
HFF.loadWeights("Weights/HierFF/weights0-4000")


# otherAgents = [EURISTIC_PLAYER, RANDOM_PLAYER, H_FF2_blue, H_GNN1_blue, H_GNN2_blue, O_FF1_blue, O_FF2_blue, O_GNN1_blue, O_GNN2_blue, H_FF1_green, H_FF2_green, H_GNN1_green, H_GNN2_green, O_FF1_green, O_FF2_green, O_GNN1_green, O_GNN2_green]
examAgent = HFF
against = RandomPlayer()
strategies = [ against, examAgent] 
gameCtrl = c.GameController.GameController(playerStrategies = strategies, idEpisode = 0, withGraphics=withGraphics, speed=True)
for i in range(0,5):
    gameCtrl.reset() 
    res = gameCtrl.playTurnamentGame()
    print(res[0])
