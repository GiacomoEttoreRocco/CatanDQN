import pandas as pd
import matplotlib.pyplot as plt
from Classes.Strategy.RLStrategyGNN import ReinforcementLearningStrategyGnn

from Classes.staticUtilities import plotCsvColumnsWithHeaders

plotCsvColumnsWithHeaders("csvFolder/results19.csv", 50, "RL-GNN", "REUR")


