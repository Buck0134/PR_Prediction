import pandas as pd

df = pd.read_csv('../data/processedData.csv')

columns_to_keep = ['prior_review_num', 'first_response_time', 'inte_open', 'account_creation_days', 'contrib_perc_commit', 'team_size', 'open_issue_num', 'project_age', 'open_pr_num', 'pr_succ_rate', 'test_lines_per_kloc', 'stars', 'integrator_availability', 'description_length', 'lifetime_minutes', 'ci_latency', 'ci_failed_perc', 'commits_on_files_touched', 'num_comments', 'first_pr', 'merged_or_not']

df = df[columns_to_keep]

df.to_csv('../data/processedDataNew.csv', index=False)

