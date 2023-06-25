import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import statistics

number_of_runs = 5

csv_filesEurVsRan = ["csvFolder/EurVsRan/results{}.csv".format(i) for i in range(1, number_of_runs)]
csv_filesHierGnnVsRan = ["csvFolder/HierGnnVsRan/results{}.csv".format(i) for i in range(1, number_of_runs)]
csv_filesHierFFVsRan = ["csvFolder/HierFFVsRan/results{}.csv".format(i) for i in range(1, number_of_runs)]
csv_filesOrchGnnVsRan = ["csvFolder/OrchGnnVsRan/results{}.csv".format(i) for i in range(1, number_of_runs)]
csv_filesOrchFFVsRan = ["csvFolder/OrchFFVsRan/results{}.csv".format(i) for i in range(1, number_of_runs)]
csv_filesRanVsRan = ["csvFolder/RanVsRan/results{}.csv".format(i) for i in range(1, number_of_runs)]

csv_filesEurVsEur = ["csvFolder/EurVsEur/results{}.csv".format(i) for i in range(1, number_of_runs)]
csv_filesHierGnnVsEur = ["csvFolder/HierGnnVsEur/results{}.csv".format(i) for i in range(1, number_of_runs)]
csv_filesHierFFVsEur = ["csvFolder/HierFFVsEur/results{}.csv".format(i) for i in range(1, number_of_runs)]
csv_filesOrchGnnVsEur = ["csvFolder/OrchGnnVsEur/results{}.csv".format(i) for i in range(1, number_of_runs)]
csv_filesOrchFFVsEur = ["csvFolder/OrchFFVsEur/results{}.csv".format(i) for i in range(1, number_of_runs)]
csv_filesRanVsEur = ["csvFolder/RanVsEur/results{}.csv".format(i) for i in range(1, number_of_runs)]


# csv_filesRanVsEur = ["csvFolder/RanVsEur/results{}.csv".format(i) for i in range(1, 11)]
# csv_filesHierGnnVsEur = ["csvFolder/HierGnnVsEur/results{}.csv".format(i) for i in range(1, 11)]
# csv_filesHierFFVsEur = ["csvFolder/HierFFVsEur/results{}.csv".format(i) for i in range(1, 11)]
# csv_filesOrchGnnVsEur = ["csvFolder/OrchGnnVsEur/results{}.csv".format(i) for i in range(1, 11)]
# csv_filesOrchFFVsEur = ["csvFolder/OrchFFVsEur/results{}.csv".format(i) for i in range(1, 11)]
# csv_filesEurVsEur = ["csvFolder/EurVsEur/results{}.csv".format(i) for i in range(1, 11)]


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
    plt.fill_between(x, q1_array, q3_array, alpha=0.2, label='Quartile interval')
    plt.xlabel('Mean of every 10 episodes (total number of episodes = 300)')
    plt.ylabel('Mean points at turn 100*')
    plt.title('Trend of values ​​with quartile range')
    plt.legend(fontsize="7", loc='lower left', bbox_to_anchor=(1, 0.5))

# ##########################################################################################################################################################

# EurVsRan = []
# HierGnnVsRan = []
# HierFFVsRan = []
# OrchGnnVsRan = []
# OrchFFVsRan = []
# RanVsRan = []


# for row in range(1, 301):
#     # print(row)
#     x = getAllfirstElements(row, 0, csv_filesEurVsRan)
#     y = getAllfirstElements(row, 0, csv_filesHierGnnVsRan)
#     z = getAllfirstElements(row, 0, csv_filesHierFFVsRan)
#     w = getAllfirstElements(row, 0, csv_filesOrchGnnVsRan)
#     a = getAllfirstElements(row, 0, csv_filesOrchFFVsRan)
#     b = getAllfirstElements(row, 0, csv_filesRanVsRan)

#     # y = getAllfirstElements(row, 0, csv_files)

#     EurVsRan.append(x)
#     HierGnnVsRan.append(y)
#     HierFFVsRan.append(z)
#     OrchGnnVsRan.append(w)
#     OrchFFVsRan.append(a)
#     RanVsRan.append(b)

# resumeValue = 10

# plot_experiment_results(riassumi(calculateRowMeans(OrchFFVsRan), resumeValue), riassumi(calculateFirstQuartiles(OrchFFVsRan), resumeValue), riassumi(calculateThirdQuartiles(OrchFFVsRan), resumeValue), "OrchestratorFF")
# plot_experiment_results(riassumi(calculateRowMeans(HierGnnVsRan), resumeValue), riassumi(calculateFirstQuartiles(HierGnnVsRan), resumeValue), riassumi(calculateThirdQuartiles(HierGnnVsRan), resumeValue), "HiearchicalGnn")
# plot_experiment_results(riassumi(calculateRowMeans(HierFFVsRan), resumeValue), riassumi(calculateFirstQuartiles(HierFFVsRan), resumeValue), riassumi(calculateThirdQuartiles(HierFFVsRan), resumeValue), "HiearchicalFF")
# plot_experiment_results(riassumi(calculateRowMeans(OrchGnnVsRan), resumeValue), riassumi(calculateFirstQuartiles(OrchGnnVsRan), resumeValue), riassumi(calculateThirdQuartiles(OrchGnnVsRan), resumeValue), "OrchestratorGnn")
# plot_experiment_results(riassumi(calculateRowMeans(EurVsRan), resumeValue), riassumi(calculateFirstQuartiles(EurVsRan), resumeValue), riassumi(calculateThirdQuartiles(EurVsRan), resumeValue), "Euristic")
# plot_experiment_results(riassumi(calculateRowMeans(RanVsRan), resumeValue), riassumi(calculateFirstQuartiles(RanVsRan), resumeValue), riassumi(calculateThirdQuartiles(RanVsRan), resumeValue), "Random")

# plt.show()

# ##########################################################################################################################################################

# EurVsEur = []
# HierGnnVsEur = []
# HierFFVsEur = []
# OrchGnnVsEur = []
# OrchFFVsEur = []
# RanVsEur = []


# for row in range(1, 301):
#     # print(row)
#     x = getAllfirstElements(row, 0, csv_filesEurVsEur)
#     y = getAllfirstElements(row, 0, csv_filesHierGnnVsEur)
#     z = getAllfirstElements(row, 0, csv_filesHierFFVsEur)
#     w = getAllfirstElements(row, 0, csv_filesOrchGnnVsEur)
#     a = getAllfirstElements(row, 0, csv_filesOrchFFVsEur)
#     b = getAllfirstElements(row, 0, csv_filesRanVsEur)

#     # y = getAllfirstElements(row, 0, csv_files)

#     EurVsEur.append(x)
#     HierGnnVsEur.append(y)
#     HierFFVsEur.append(z)
#     OrchGnnVsEur.append(w)
#     OrchFFVsEur.append(a)
#     RanVsEur.append(b)

# resumeValue = 10

# plot_experiment_results(riassumi(calculateRowMeans(OrchFFVsEur), resumeValue), riassumi(calculateFirstQuartiles(OrchFFVsEur), resumeValue), riassumi(calculateThirdQuartiles(OrchFFVsEur), resumeValue), "OrchestratorFF")
# plot_experiment_results(riassumi(calculateRowMeans(HierGnnVsEur), resumeValue), riassumi(calculateFirstQuartiles(HierGnnVsEur), resumeValue), riassumi(calculateThirdQuartiles(HierGnnVsEur), resumeValue), "HiearchicalGnn")
# plot_experiment_results(riassumi(calculateRowMeans(HierFFVsEur), resumeValue), riassumi(calculateFirstQuartiles(HierFFVsEur), resumeValue), riassumi(calculateThirdQuartiles(HierFFVsEur), resumeValue), "HiearchicalFF")
# plot_experiment_results(riassumi(calculateRowMeans(OrchGnnVsEur), resumeValue), riassumi(calculateFirstQuartiles(OrchGnnVsEur), resumeValue), riassumi(calculateThirdQuartiles(OrchGnnVsEur), resumeValue), "OrchestratorGnn")
# plot_experiment_results(riassumi(calculateRowMeans(EurVsEur), resumeValue), riassumi(calculateFirstQuartiles(EurVsEur), resumeValue), riassumi(calculateThirdQuartiles(EurVsEur), resumeValue), "Euristic")
# plot_experiment_results(riassumi(calculateRowMeans(RanVsEur), resumeValue), riassumi(calculateFirstQuartiles(RanVsEur), resumeValue), riassumi(calculateThirdQuartiles(RanVsEur), resumeValue), "Random")

# plt.show()
csv_fileHigh = ["csvFolder/HighTrainedHierGnn/results0.csv"]

with open("csvFolder/HighTrainedHierGnn/results0.csv", 'r') as file:
    # Leggi il file CSV
    reader = csv.reader(file)
    
    # Salta la prima riga del file
    next(reader)
    
    # Inizializza una lista vuota per la prima colonna
    high = []
    
    # Itera sulle righe del file CSV
    for row in reader:
        # Aggiungi il valore della prima colonna come float alla lista
        high.append(float(row[0]))
resumeValue = 10

print(high)

plt.plot(riassumi(high, 50))

# plot_experiment_results(riassumi(calculateRowMeans(high), resumeValue), riassumi(calculateFirstQuartiles(high), resumeValue), riassumi(calculateThirdQuartiles(high), resumeValue), "high")

plt.show()
