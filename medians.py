import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_files = ["csvFolder/results{}.csv".format(seed) for seed in range(1, 20)]

first_column_medians = []
first_column_first_quartiles = []
first_column_third_quartiles = []

second_column_medians = []
second_column_first_quartiles = []
second_column_third_quartiles = []

for file in csv_files:
    df = pd.read_csv(file)
    first_column = df.iloc[:, 0]
    second_column = df.iloc[:, 1]
    
    first_column_medians.append(first_column.median())
    first_column_first_quartiles.append(first_column.quantile(0.25))
    first_column_third_quartiles.append(first_column.quantile(0.75))
    
    second_column_medians.append(second_column.median())
    second_column_first_quartiles.append(second_column.quantile(0.25))
    second_column_third_quartiles.append(second_column.quantile(0.75))

fig, ax = plt.subplots()

x = np.arange(1, 21)

ax.plot(x, first_column_medians, label='Prima Colonna - Mediana')
ax.fill_between(x, first_column_first_quartiles, first_column_third_quartiles, alpha=0.2)

ax.plot(x, second_column_medians, label='Seconda Colonna - Mediana')
ax.fill_between(x, second_column_first_quartiles, second_column_third_quartiles, alpha=0.2)

ax.set_xlabel('CSV')
ax.set_ylabel('Valore')
ax.set_title('Mediana e Quartili per Colonna')

ax.legend()
plt.show()