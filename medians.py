import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import statistics

csv_filesEurVsRan = ["csvFolder/EurVsRan/results{}.csv".format(i) for i in range(1, 11)]
csv_filesHierGnnVsRan = ["csvFolder/HierGnnVsRan/results{}.csv".format(i) for i in range(1, 11)]
csv_filesHierFFVsRan = ["csvFolder/HierFFVsRan/results{}.csv".format(i) for i in range(1, 11)]
csv_filesOrchGnnVsRan = ["csvFolder/OrchGnnVsRan/results{}.csv".format(i) for i in range(1, 11)]
csv_filesOrchFFVsRan = ["csvFolder/OrchFFVsRan/results{}.csv".format(i) for i in range(1, 11)]

# csv_filesRanVsEur = ["csvFolder/RanVsEur/results{}.csv".format(i) for i in range(1, 11)]
# csv_filesGnnHierVsEur = ["csvFolder/GnnHierVsEur/results{}.csv".format(i) for i in range(1, 11)]
# csv_filesHierFFVsEur = ["csvFolder/HierFFVsEur/results{}.csv".format(i) for i in range(1, 11)]
# csv_filesOrchGnnVsEur = ["csvFolder/OrchGnnVsEur/results{}.csv".format(i) for i in range(1, 11)]
# csv_filesOrchFFVsEur = ["csvFolder/OrchFFVsEur/results{}.csv".format(i) for i in range(1, 11)]

def getAllfirstElements(row_index, column_index, csv_files):
    first_elements = []
    for file in csv_files:
        with open(file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if row_index < len(rows):
                row = rows[row_index]
                first_element = float(row[column_index])
                first_elements.append(first_element)
    return first_elements

def plotMeansColumnsWithHeaders(colonna1, colonna2, interval, nome1, nome2):
    # Calcola i punti medi ogni interval punti
    punti_medi_colonna1 = []
    punti_medi_colonna2 = []
    for i in range(0, len(colonna1), interval):
        if i + interval <= len(colonna1):
            media_colonna1 = sum(colonna1[i:i+interval]) / float(interval)
            media_colonna2 = sum(colonna2[i:i+interval]) / float(interval)
        else:
            media_colonna1 = sum(colonna1[i:]) / float(len(colonna1) - i)
            media_colonna2 = sum(colonna2[i:]) / float(len(colonna2) - i)
        punti_medi_colonna1.append(media_colonna1)
        punti_medi_colonna2.append(media_colonna2)

    # Crea un array di indici per i punti medi
    indici_punti_medi = range(0, len(colonna1), interval)

    # Traccia i punti medi come line plot
    plt.plot(indici_punti_medi, punti_medi_colonna1, color='red', label='Media ogni ' + str(interval) + ' punti (' + nome1 + ')')
    plt.plot(indici_punti_medi, punti_medi_colonna2, color='blue', label='Media ogni ' + str(interval) + ' punti (' + nome2 + ')')

    # Imposta i titoli degli assi e la legenda
    plt.xlabel('Indice')
    plt.ylabel('Valore')
    plt.legend()

    # Mostra il grafico
    plt.show()
# print(get_first_elements(2, 1, csv_files))
# plotMeansColumnsWithHeaders(meansGNN, meansRAN, 15, "x", "y")

def riassumi(data, interval):
    # Calcola i punti medi ogni interval punti
    punti_medi_data = []
    for i in range(0, len(data), interval):
        if i + interval <= len(data):
            media_data = sum(data[i:i+interval]) / float(interval)
        else:
            media_data = sum(data[i:]) / float(len(data) - i)
        punti_medi_data.append(media_data)
    return punti_medi_data

def calculateRowMeans(matrix):
    matrix = np.array(matrix)
    row_means = np.mean(matrix, axis=1)
    return row_means

def calculateFirstQuartiles(matrix):
    matrix = np.array(matrix)
    row_quartiles = np.percentile(matrix, q=25, axis=1)
    return row_quartiles

def calculateThirdQuartiles(matrix):
    matrix = np.array(matrix)
    row_quartiles = np.percentile(matrix, q=75, axis=1)
    return row_quartiles

def plot_experiment_results(mean_array, q1_array, q3_array, name):
    x = np.arange(1, len(mean_array) + 1)
    plt.ylim(2, 11)
    plt.plot(x, mean_array, label='Mean ' + name)
    plt.fill_between(x, q1_array, q3_array, alpha=0.3, label='Quartile interval')
    plt.xlabel('Mean of every 5 episodes (total number of episodes = 1000)')
    plt.ylabel('Mean points at turn 100*')
    plt.title('Trend of values ​​with quartile range')
    plt.legend(fontsize="7", loc='lower left')
    # plt.show()
# meansGNN = []
# meansRAN = []

EurVsRan = []
HierGnnVsRan = []

for row in range(1, 301):
    # print(row)
    x = getAllfirstElements(row, 0, csv_filesEurVsRan)
    y = getAllfirstElements(row, 0, csv_filesHierGnnVsRan)
    EurVsRan.append(x)
    HierGnnVsRan.append(y)

# print(np.mean(gnnRes[1]))
# print(np.mean(ranRes[1]))
# print(calculateRowMeans(gnnRes))

# print(len(gnnRes))
# plot_experiment_results(calculateRowMeans(gnnRes), calculateFirstQuartiles(gnnRes), calculateThirdQuartiles(gnnRes))
# plot_experiment_results(calculateRowMeans(ranRes), calculateFirstQuartiles(ranRes), calculateThirdQuartiles(ranRes))

plot_experiment_results(riassumi(calculateRowMeans(EurVsRan), 3), riassumi(calculateFirstQuartiles(EurVsRan), 3), riassumi(calculateThirdQuartiles(EurVsRan), 3), "Euristic")
plot_experiment_results(riassumi(calculateRowMeans(HierGnnVsRan), 3), riassumi(calculateFirstQuartiles(HierGnnVsRan), 3), riassumi(calculateThirdQuartiles(HierGnnVsRan), 3), "HiearchicalGnn")

plt.show()
# plt.show()
