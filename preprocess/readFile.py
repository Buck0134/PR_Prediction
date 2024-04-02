import pandas as pd
import numpy as np


class DataCleaner:
    def __init__(self, file_path="../data/new_pullreq.csv"):
        self.df = pd.read_csv(file_path)

    def df_shape(self):
        return self.df.shape

    def df_head(self, n=5):
        pd.set_option("display.max_rows", None)
        return self.df['ci_latency'].head(n)

    # def pre_process(self):
    #     # Compute the boolean mask for NaN values in each row
    #     # numerical --> mean
    #     # dummy --> Missing
    #     # categorical data --> Mode
    #     nan_mask = self.df.isnull().any(axis=1)

    #     # Filter the DataFrame to get rows without any NaN values
    #     clean_df = self.df[~nan_mask]

    #     # Check if every row has non-NaN values
    #     all_rows_are_complete = clean_df.shape[0] == self.df.shape[0]

    #     if all_rows_are_complete:
    #         print("All rows are complete.")
    #     else:
    #         print(
    #             "Some rows contain NaN values and have been removed. Some of the missings values are recreated"
    #         )
    def pre_process(self):
        # Fill missing values for each column based on its type
        for column in self.df.columns:
            # Detect and process dummy columns
            if (
                self.df[column].dtype == object
                # self.df[column].dtype == np.object
                or self.df[column].dtype.name == "category"
            ):
                if (
                    self.df[column].nunique() == 2
                    or self.df[column].dtype.name == "category"
                ):
                    self.df[column] = (
                        self.df[column].fillna("Missing").astype("category")
                    )  # fillna
                else:
                    # Process categorical data: Fill with the most frequent value
                    most_frequent = self.df[column].mode()[0]
                    self.df[column] = self.df[column].fillna(most_frequent)
            else:
                # Process numerical data: Fill with the mean value
                self.df[column] = self.df[column].fillna(self.df[column].mean())

        print("Data pre-processing completed.")

    def find_columns_with_all_data(self):
        # Compute the sum of missing values for each column
        missing_values_sum = self.df.isnull().sum()

        # Filter out columns where the sum of missing values is zero
        columns_with_all_data = missing_values_sum[
            missing_values_sum == 0
        ].index.tolist()

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

    def find_columns_with_excessive_missing_data(self):
        total_rows = self.df.shape[0]
        missing_percentage = (self.df.isnull().sum() / total_rows) * 100
        columns_excessive_missing = missing_percentage[
            missing_percentage > 60
        ].index.tolist()
        if columns_excessive_missing:
            print("Columns with more than 50% missing data:")
            print(columns_excessive_missing)
        else:
            print("No columns have more than 50% missing data.")

    def display_categorical_missing_data_percentage(self):
        pd.set_option("display.max_rows", None)
        categorical_columns = self.df.select_dtypes(include=["object"]).columns
        # Calculate the percentage of missing data for each column
        missing_percentage = (
            self.df[categorical_columns].isnull().sum() / len(self.df) * 100
        )

        # Display the results
        print("Percentage of missing data per categorical column:")
        print(missing_percentage)

    def display_missing_data_percentage(self):
        pd.set_option("display.max_rows", None)
        # Calculate the percentage of missing data for each column
        missing_percentage = (
            self.df.isnull().sum() / len(self.df) * 100
        )
        # Display the results
        print("Percentage of missing data per column:")
        print(missing_percentage)

    def drop_columns_with_excessive_missing_data(self):
        # Calculate the percentage of missing data for each column
        missing_percentage = self.df.isnull().sum() / len(self.df) * 100

        # Identify columns where more than 50% of the data is missing
        columns_to_drop = missing_percentage[missing_percentage > 60].index

        # Drop these columns from the DataFrame
        self.df.drop(columns=columns_to_drop, inplace=True)

        # Drop the column: inte_first_emo because hard to retrieve with actual user data
        self.df.drop("inte_first_emo", axis=1, inplace=True)

        # Print the names of the dropped columns for confirmation
        if len(columns_to_drop) > 0:
            print("Dropped columns with more than 60% missing data:")
            print(columns_to_drop.tolist())
        else:
            print("No columns have more than 60% missing data.")
    
    def getPreprocessedData():
        pass

    def print_rows_based_on_conditions(df):
    # Check if both 'ci_exists' and 'ci_latency' columns are present in the DataFrame
        if 'ci_exists' in df.columns and 'ci_latency' in df.columns:
        # Filter rows where 'ci_exists' is False and 'ci_latency' is not null
            filtered_rows = df[(df['ci_exists'] == False) & (df['ci_latency'].notnull())]
        
        # Check if there are any rows to display
            if not filtered_rows.empty:
                print("Rows where 'ci_exists' is False and 'ci_latency' is not null:")
                print(filtered_rows)
            else:
                print("No rows match the criteria.")
        else:
            print("Required columns ('ci_exists' and/or 'ci_latency') are missing in the DataFrame.")


# Example usage:
cleaner = DataCleaner()
print("Before dropping columns, shape:", cleaner.df_shape())
# === STEP 1 - Drop Columns =====
cleaner.drop_columns_with_excessive_missing_data()
print("After dropping columns, shape:", cleaner.df_shape())
# === STEP 1 DONE =====
print("Pre")
print(cleaner.df_head())
print(cleaner.display_missing_data_percentage())
print("Processing")
cleaner.pre_process()
print("After")
print(cleaner.df_head())
print(cleaner.display_missing_data_percentage())
