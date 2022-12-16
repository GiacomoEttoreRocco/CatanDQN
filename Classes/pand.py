import pandas as pd
d = {'owner':  [4,5,6], 'isCity': [0,0,1]}
df = pd.DataFrame(data=d)

d1 = {'owner':  [4,6], 'isCity': [1,1]}
df1 = pd.DataFrame(data=d1)

d2 = {'player':  [4], 'ale': [0], 'giac':[4], 'winner':[None]}
df2 = pd.DataFrame(data=d2)

d3 = {'player':  [2], 'ale': [0], 'giac':[10], 'winner':[None]}
df3 = pd.DataFrame(data=d3)

dt = {'places': [], 'glob':[]}
whole = pd.DataFrame(data=dt, dtype=object)

whole.loc[len(whole)] = [d, d2]


whole.loc[len(whole)] = [d1, d3]

print(whole)

whole.to_json("C:/Users/ricca/OneDrive/Documents/Università/CatanGNN-AI/csv/trial.json")




# df = pd.read_csv("C:/Users/ricca/OneDrive/Documents/Università/CatanGNN-AI/csv/trial.csv")

# print(pd.read_table("C:/Users/ricca/OneDrive/Documents/Università/CatanGNN-AI/csv/trial.csv"))

