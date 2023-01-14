import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results_ric.csv", names=['idx', '1_prior_vs_hyb', '2_prior_vs_hyb', 'prior_vs_hyb', '1_prior_vs_pure', '2_prior_vs_pure', 'prior_vs_pure', '1_hyb_vs_pure', '2_hyb_vs_pure', 'hyb_vs_pure'], header=0)

print(df)
plt.plot(df['prior_vs_hyb'], label='prior_vs_hyb')
plt.plot(df['prior_vs_pure'], label='prior_vs_pure')
plt.plot(df['hyb_vs_pure'], label='hyb_vs_pure')
plt.legend()
plt.show()

