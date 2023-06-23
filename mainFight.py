
from Classes.Strategy.RLStrategyFFhier import ReinforcementLearningStrategyFfHier
from Classes.Strategy.RanPlayer import RandomPlayer
import Classes as c

withGraphics = True

def doGame(agent1, agent2, path1, path2):
    strategies = [rlStrategyFfHier, randomPlayer]
    gameCtrl = c.GameController.GameController(playerStrategies = strategies, idEpisode = 0, withGraphics=withGraphics, speed=True)
    if("RL" in agent1.name()):
        agent1 = agent1.loadWeights(path1)
    if("RL" in agent2.name()):
        agent2 = agent2.loadWeights(path2)
    return gameCtrl.playTurnamentGame()


rlStrategyFfHier = ReinforcementLearningStrategyFfHier()
randomPlayer = RandomPlayer()

print(rlStrategyFfHier.name())
print(randomPlayer.name())
# doGame(rlStrategyFfHier, randomPlayer, "Weights/HierFFVsRan/weights"+str(1), "")