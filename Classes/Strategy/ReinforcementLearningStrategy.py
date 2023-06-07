from Classes.Strategy.Strategy import Strategy

class ReinforcementLearningStrategy(Strategy):
    def name(self):
        return "RL"

    def bestAction(self, player):
        return super().bestAction(player)