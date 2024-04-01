import pandas as pd

file_path = '../data/new_pullreq.csv'

df = pd.read_csv(file_path)

print(df.head())
