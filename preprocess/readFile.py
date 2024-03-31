import pandas as pd

file_path = '/Users/liqiu/PR_Prediction/data/new_pullreq.csv'

df = pd.read_csv(file_path)

print(df.head())
print(df.shape)