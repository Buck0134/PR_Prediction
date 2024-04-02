import pandas as pd
import numpy as np

# TODO:
# drop ci_exists
# double check consistency

conditions = {
    "has_comments": ["perc_pos_emotion", "perc_neg_emotion", "first_response_time"],
    "contrib_comment": ["perc_contrib_pos_emo", "perc_contrib_neg_emo"],
    "inte_comment": ["perc_inte_neg_emo", "perc_inte_pos_emo"],
    "ci_exists": ["ci_latency", "ci_failed_perc"],
    "same_user": [
        "same_country",
        "same_affiliation",
        "contrib_follow_integrator",
        "open_diff",
        "cons_diff",
        "extra_diff",
        "agree_diff",
        "neur_diff",
    ],
}


class DataCleaner:
    def __init__(self, file_path="../data/new_pullreq.csv"):
        self.df = pd.read_csv(file_path)

    def df_shape(self):
        return self.df.shape

    def df_head(self, n=5):
        pd.set_option("display.max_rows", None)
        return self.df.head(n)
    
    def drop_rows_with_missing_ci_exists(self):
        initial_shape = self.df.shape
        self.df = self.df.dropna(subset=['ci_exists'])
        final_shape = self.df.shape
    
        print(f"Rows dropped: {initial_shape[0] - final_shape[0]}")
        print(f"New DataFrame shape: {final_shape}")

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

    def apply_pre_post_conditions(self, pre_condition_col, post_condition_cols):
        """
        Modify DataFrame based on a precondition and postconditions.

        Parameters:
        - df: pandas DataFrame.
        - pre_condition_col: The column name for the precondition.
        - post_condition_cols: A list of column names for the postconditions.

        If the precondition of a row is False, all specified postcondition columns in that row are set to NaN.
        """
        # Ensure post_condition_cols is a list to allow iteration
        if not isinstance(post_condition_cols, list):
            raise ValueError("post_condition_cols must be a list of column names")

        # Apply the precondition logic and update postcondition columns where necessary
        for index, row in self.df.iterrows():
            if (
                not row[pre_condition_col]
                or row[pre_condition_col] == 0
                or row[pre_condition_col] == "0"
            ):  # If the precondition is False
                for col in post_condition_cols:
                    self.df.at[index, col] = (
                        np.nan
                    )  # Set the postcondition columns to NaN

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
        missing_percentage = self.df.isnull().sum() / len(self.df) * 100
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

    def print_rows_based_on_conditions(df):
        # Check if both 'ci_exists' and 'ci_latency' columns are present in the DataFrame
        if "ci_exists" in df.columns and "ci_latency" in df.columns:
            # Filter rows where 'ci_exists' is False and 'ci_latency' is not null
            filtered_rows = df[
                (df["ci_exists"] == False) & (df["ci_latency"].notnull())
            ]

            # Check if there are any rows to display
            if not filtered_rows.empty:
                print("Rows where 'ci_exists' is False and 'ci_latency' is not null:")
                print(filtered_rows)
            else:
                print("No rows match the criteria.")
        else:
            print(
                "Required columns ('ci_exists' and/or 'ci_latency') are missing in the DataFrame."
            )

    def getDFPreprocessedData(self):
        self.drop_columns_with_excessive_missing_data()
        self.drop_rows_with_missing_ci_exists()
        self.pre_process()
        for key, value in conditions.items():
            self.apply_pre_post_conditions(key, value)
        return self.df

    def createCSVPreProcessData(self, file_path = "../data/processedData.csv"):
        self.df.to_csv(file_path, index=False)
        print(f"DataFrame exported to {file_path}.")


# Example usage:
cleaner = DataCleaner()
# false_count = (cleaner.df["has_comments"] == False).sum()
# total_count = len(cleaner.df["has_comments"])
# percentage_false = (false_count / total_count) * 100
# print(f"Percentage of False values: {percentage_false}%")
# print("Before dropping columns, shape:", cleaner.df_shape())
# === STEP 1 - Drop Columns =====
cleaner.drop_columns_with_excessive_missing_data()
print("After dropping columns, shape:", cleaner.df_shape())
# === STEP 1 DONE =====
print("Pre")
# print(cleaner.df_head())
print("dropping ci_exists rows")
cleaner.drop_rows_with_missing_ci_exists()
print("===========================================")
print(cleaner.display_missing_data_percentage())
print("Processing")
cleaner.pre_process()
# cleaner.apply_pre_post_conditions("ci_exists", ["ci_latency", "ci_failed_perc"])
for key, value in conditions.items():
    cleaner.apply_pre_post_conditions(key, value)
print("Conditions check done")
print("After")
# print(cleaner.df_head())
print(cleaner.display_missing_data_percentage())
