import pandas as pd

class DataCleaner:
    def __init__(self, file_path='../data/new_pullreq.csv'):
        self.df = pd.read_csv(file_path)
    
    def df_shape(self):
        return self.df.shape
    
    def df_head(self, n=5):
        return self.df.head(n)
    
    def check_complete_rows(self):
        # Compute the boolean mask for NaN values in each row
        nan_mask = self.df.isnull().any(axis=1)
        
        # Filter the DataFrame to get rows without any NaN values
        clean_df = self.df[~nan_mask]
        
        # Check if every row has non-NaN values
        all_rows_are_complete = clean_df.shape[0] == self.df.shape[0]
        
        if all_rows_are_complete:
            print("All rows are complete.")
        else:
            print("Some rows contain NaN values and have been removed.")
    
    def find_columns_with_all_data(self):
        # Compute the sum of missing values for each column
        missing_values_sum = self.df.isnull().sum()
        
        # Filter out columns where the sum of missing values is zero
        columns_with_all_data = missing_values_sum[missing_values_sum == 0].index.tolist()
        
        if columns_with_all_data:
            print("Columns with all rows containing data:")
            print(columns_with_all_data)
        else:
            print("No columns have all rows containing data.")
    
    def find_columns_with_missing_values(self, row_start, row_end):
        # Check for NaN values in specified rows
        nan_rows = self.df.iloc[row_start:row_end].isnull()
        
        # Sum NaN values for each column
        nan_sum = nan_rows.sum()
        
        # Find columns with missing values
        columns_with_missing_values = nan_sum[nan_sum > 0].index.tolist()
        
        if columns_with_missing_values:
            print("Columns with missing values in rows", row_start, "to", row_end)
            print(columns_with_missing_values)
        else:
            print("No columns have missing values in rows", row_start, "to", row_end)

# Example usage:
cleaner = DataCleaner()
print("DataFrame shape:", cleaner.df_shape())
print("\nDataFrame head:")
print(cleaner.df_head())
cleaner.check_complete_rows()
cleaner.find_columns_with_all_data()
cleaner.find_columns_with_missing_values(1, 10)
