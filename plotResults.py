import pandas as pd
import matplotlib.pyplot as plt
from Classes.Strategy.RLStrategyGNN import ReinforcementLearningStrategyGnn

from Classes.staticUtilities import plotCsvColumnsWithHeaders

plotCsvColumnsWithHeaders("csvFolder/results9.csv", 70, "RL-GNN", "REUR")


