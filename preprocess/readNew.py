import pandas as pd

file_path = "../data/processedData.csv"

df = pd.read_csv(file_path)

print(df.shape)

file_path_old = "../data/new_pullreq.csv"

df_old = pd.read_csv(file_path_old)

print(df_old.shape)