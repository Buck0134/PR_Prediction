import pandas as pd
pd.set_option('display.max_columns', None)
df = pd.read_csv('../data/processedDataNew.csv')

print(df.head())