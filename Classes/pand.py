import pandas as pd
d = {'owner':  [4,5,6], 'isCity': [0,0,1]}
df = pd.DataFrame(data=d)

d1 = {'cicco':  [4], 'pasticcio': [0], 'pollo':[4]}
df1 = pd.DataFrame(data=d1)

d2 = {'player':  [4], 'ale': [0], 'giac':[4], 'winner':[None]}
df2 = pd.DataFrame(data=d2)

d3 = {'asd':  [4], 'fgh': [0], 'qwe':[4]}
df2 = pd.DataFrame(data=d2)

dt = {'places': [df1, df2], 'edges':[df3, df4], 'glob':[df5.iloc[0],df6.iloc[0]]}
whole = pd.DataFrame(data=dt)
wholeGlob = pd.DataFrame(whole.glob)

# print(whole.iloc[1])
print(wholeGlob)
# whole.glob.winner.replace(to_replace = 'None', value = 1)



