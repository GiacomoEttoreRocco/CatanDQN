from Classes.Strategy.Strategy import Strategy
from RL.DQN import DQNagent

class ReinforcementLearningStrategy(Strategy):
    def __init__(self): # diventerà un singleton
        # self, nInputs, nOutputs, criterion, device
        self.macroDQN = DQNagent(nInputs, nOutputs, criterion, device) # macro rete decisionale

    def name(self):
        return "RL"

    def bestAction(self, player):
        # psuedo codice:
        stato = player.game.getState(player)

        # outputs = DQN(player.state, player, self.player._victoryPoints)
        outputs = self.DQN.selectMove(player.state, self.player._victoryPoints, mask)


        # action, thingNeeded = prendiIlMassimoTraLeAzioniEseguibili(outputs)
        # action, thingNeeded = DQN.greedyAction(outputs, mask)

        return action, thingNeeded, None