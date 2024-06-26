import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time

class PullRequestPredictor:
    def __init__(self, df):
        self.df = df
        self.top_numerical_features = []
        self.top_categorical_features = []
        self.class_distribution = {}

    def preprocess(self):
        # Remove columns as per your specification
        columns_to_remove = ['project_id', 'test_cases_per_kloc', 'asserts_per_kloc', 'churn_addition', 'churn_deletion', 'at_tag', 
                            'num_participants', 'num_comments_con', 'perc_external_contribs', 'social_strength', 
                            'requester_succ_rate', 'fork_num', 'prior_interaction', 'part_num_code', 
                            'num_code_comments_con', 'fork_num', 'ci_build_num', 'has_participants', 'core_comment', 
                            'contrib_comment', 'inte_comment', 'has_exchange', 'contrib_country', 
                            'perc_contrib_neg_emo', 'contrib_first_emo', 'ci_test_passed', 'ci_first_build_status', 
                            'ci_last_build_status', 'perc_contrib_pos_emo', 'perc_inte_neg_emo', 'perc_inte_pos_emo',
                            'contrib_follow_integrator', 'same_country', 'same_affiliation', 'contrib_gender', 
                            'contrib_country', 'id', 'creator_id', 'last_closer_id', 'last_close_time', 
                            'language', 'same_user', 'open_diff', 'cons_diff', 'extra_diff', 'agree_diff', 'neur_diff', 
                            'perc_neg_emotion', 'perc_pos_emotion', 'core_member', 'contrib_open', 'contrib_neur', 'contrib_extra',
                            'contrib_agree', 'inte_open', 'inte_cons', 'inte_extra', 'inte_agree', 'inte_neur', 'sloc', 'test_lines_per_kloc', 'asserts_per_kloc', 'integrator_availability']
        self.df.drop(columns=columns_to_remove, inplace=True, errors='ignore')
        
        self.categorical_features = [col for col in self.df.columns if self.df[col].dropna().isin([0, 1]).all() and col != 'merged_or_not' and col != 'project_id']
        self.numerical_features = [col for col in self.df.columns if col not in self.categorical_features + ['merged_or_not', 'project_id']]
        
        # Impute and scale numerical features
        columns_with_nan = self.df.columns[self.df.isna().any()].tolist()
        for column in columns_with_nan:
            average = self.df[column].mean()
            if column == 'first_response_time':
                self.df[column] = self.df.apply(lambda row: 0 if row['has_comments'] == 0 else average, axis=1)
            if column == 'ci_latency' or column == 'ci_failed_perc':
                self.df[column] = self.df.apply(lambda row: 0 if row['ci_exists'] == 0 else average, axis=1)
        
        columns_with_nan = self.df.columns[self.df.isna().any()].tolist()
        assert len(columns_with_nan) == 0, f"Columns with NaN: {columns_with_nan}"
        
        scaler = MinMaxScaler()
        self.df[self.numerical_features] = scaler.fit_transform(self.df[self.numerical_features])
        
        same_value_1 = (self.df['ci_latency'] == self.df['ci_exists']).all()
        same_value_2 = (self.df['ci_failed_perc'] == self.df['ci_exists']).all()
        same_value_3 = (self.df['first_response_time'] == self.df['has_comments']).all()
        print(f"After imputing nan, column ci_latency and ci_exists are the same? {same_value_1}")
        print(f"After imputing nan, column ci_failed_perc and ci_exists are the same? {same_value_2}")
        print(f"After imputing nan, column first_response_time and has_comments are the same? {same_value_3}")
        
        # Remove column first_response_time and ci_latency
        # self.df.drop(columns=['first_response_time', 'ci_latency'])

        self.class_distribution['Class 0'] = len(self.df[self.df['merged_or_not'] == 0])
        self.class_distribution['Class 1'] = len(self.df[self.df['merged_or_not'] == 1])
        print(f"Number of Class 0: {self.class_distribution['Class 0']}, Number of Class 1: {self.class_distribution['Class 1']}")

    def select_features(self, N, M):
        
        self.start_time = time.time()
        
        self.N = N
        self.M = M
        # Reset top features
        self.top_numerical_features = []
        self.top_categorical_features = []
        
        # For Numerical Features
        if self.N > 0:
            selector_numerical = SelectKBest(score_func=f_classif, k=N)
            selector_numerical.fit(self.df[self.numerical_features], self.df['merged_or_not'])
            self.top_numerical_features = [self.numerical_features[i] for i in selector_numerical.get_support(indices=True)]

        # For Categorical Features
        if self.M > 0:
            selector_categorical = SelectKBest(score_func=chi2, k=M)
            selector_categorical.fit(self.df[self.categorical_features], self.df['merged_or_not'])
            self.top_categorical_features = [self.categorical_features[i] for i in selector_categorical.get_support(indices=True)]

    def fit_and_evaluate(self):
        try:
            features = self.top_numerical_features + self.top_categorical_features
            features.append('merged_or_not')
            
            df_train, df_test = train_test_split(self.df[features], test_size=0.2, random_state=42)

            clf = DecisionTreeClassifier(max_depth = 20)

            clf.fit(df_train[self.top_numerical_features + self.top_categorical_features], df_train['merged_or_not'])

            predictions = clf.predict(df_test[self.top_numerical_features + self.top_categorical_features])

            accuracy = accuracy_score(df_test['merged_or_not'], predictions)
            f1 = f1_score(df_test['merged_or_not'], predictions)
            precision = precision_score(df_test['merged_or_not'], predictions)
            recall = recall_score(df_test['merged_or_not'], predictions)

            elapsed_time = time.time() - self.start_time

            return {'N': self.N, 'M': self.M, 'Precision': precision, 'Recall': recall, 'F1': f1, 'Accuracy': accuracy, 'Time': elapsed_time, 'Features': features}
        except Exception as e:
            print(e)
            return {'N': self.N, 'M': self.M, 'Precision': 0, 'Recall': 0, 'F1': 0, 'Accuracy': 0, 'Time': 0, 'Features': features}