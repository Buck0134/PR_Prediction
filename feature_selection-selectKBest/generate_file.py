import pandas as pd

df = pd.read_csv('data/processedData.csv')

columns_to_keep = ['first_response_time', 'account_creation_days', 'contrib_perc_commit', 'team_size', 'open_issue_num', 'project_age', 'open_pr_num', 'pr_succ_rate', 'stars', 'description_length', 'lifetime_minutes', 'ci_latency', 'ci_failed_perc', 'commits_on_files_touched', 'num_comments', 'merged_or_not']

df = df[columns_to_keep]

df.to_csv('data/processedDataNew.csv', index=False)

