import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results_jack_01.csv", names=['idx', '1_prior_vs_hyb', '2_prior_vs_hyb', 'prior_vs_hyb', '1_prior_vs_pure', '2_prior_vs_pure', 'prior_vs_pure', '1_hyb_vs_pure', '2_hyb_vs_pure', 'hyb_vs_pure'], header=None)
# df = pd.read_csv("results_jack_01.csv", names=['idx', 'prior_vs_hyb', 'prior_vs_pure', 'hyb_vs_pure'], header=None)

print(df)
plt.plot(df['prior_vs_hyb'], label='3 priority vs 1 hybrid')
plt.plot(df['prior_vs_pure'], label='3 priority vs 1 pure')
plt.plot(df['hyb_vs_pure'], label='3 hybrid vs 1 pure')
# plt.title("Sports Watch Data")
plt.xlabel("Iterations")
plt.ylabel("Win rate for evaluated player")
plt.legend()
plt.show()


df = pd.read_csv("best_results.csv", names=['prior_vs_hyb', 'prior_vs_pure', 'hyb_vs_pure'], header=None)
data = [df['prior_vs_hyb'], df['prior_vs_pure'], df['hyb_vs_pure']]
fig, ax = plt.subplots()
ax.set_title('Best weights performance')
ax.set_ylabel("Win rate")
ax.set_xticklabels(['Priority vs Hybrid', 'Priority vs Pure', 'Hybrid vs Pure'])
ax.boxplot(data)

plt.show()