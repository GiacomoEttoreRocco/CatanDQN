import pandas as pd
import matplotlib.pyplot as plt

from Classes.staticUtilities import plotCsvColumnsWithHeaders

plotCsvColumnsWithHeaders("csvFolder/results.csv", 5, "RL-GNN", "REUR")
