import pandas as pd


def filter_data():
    print("Filtering data...")
    df = pd.read_csv("data/processedData.csv")
    columns_to_remove = [
        "project_id",
        "test_cases_per_kloc",
        "asserts_per_kloc",
        "churn_addition",
        "churn_deletion",
        "at_tag",
        "num_participants",
        "num_comments_con",
        "perc_external_contribs",
        "social_strength",
        "requester_succ_rate",
        "fork_num",
        "prior_interaction",
        "part_num_code",
        "num_code_comments_con",
        "fork_num",
        "ci_build_num",
        "has_participants",
        "core_comment",
        "contrib_comment",
        "inte_comment",
        "has_exchange",
        "contrib_country",
        "contrib_first_emo",
        "perc_contrib_neg_emo",
        "contrib_first_emo",
        "ci_test_passed",
        "ci_first_build_status",
        "ci_last_build_status",
        "perc_contrib_pos_emo",
        "perc_inte_neg_emo",
        "perc_inte_pos_emo",
        "contrib_follow_integrator",
        "same_country",
        "same_affiliation",
        "contrib_gender",
        "contrib_country",
        "id",
        "creator_id",
        "last_closer_id",
        "last_close_time",
        "language",
        "same_user",
        "open_diff",
        "cons_diff",
        "extra_diff",
        "agree_diff",
        "neur_diff",
        "perc_neg_emotion",
        "perc_pos_emotion",
    ]
    df.drop(columns=columns_to_remove, inplace=True, errors="ignore")
    df.fillna(0, inplace=True)
    cols = df.columns.tolist()  # Get a list of all columns
    cols.append(
        cols.pop(cols.index("merged_or_not"))
    )  # Remove 'merged_or_not' and add it to the end
    df.to_csv("data/filteredData.csv", index=False)


def generate_data_GA():
    print("Generating data after GA feature selection...")
    df = pd.read_csv("data/filteredData.csv")
    df = df[
        [
            "merged_or_not",
            "first_pr",
            "contrib_cons",
            "followers",
            "account_creation_days",
            "open_pr_num",
            "pushed_delta",
            "pr_succ_rate",
            "stars",
            "description_length",
            "test_inclusion",
            "num_code_comments",
            "test_churn",
            "ci_failed_perc",
            "src_churn",
            "files_added",
            "files_deleted",
            "files_changed",
            "has_comments",
            "num_comments",
            "other_comment",
        ]
    ]
    df.to_csv("data/filteredData_selected_GA.csv", index=False)


def generate_data_pso():
    print("Generating data after PSO feature selection...")
    df = pd.read_csv("data/filteredData.csv")
    df = df[
        [
            "merged_or_not",
            "first_pr",
            "prior_review_num",
            "first_response_time",
            "followers",
            "prev_pullreqs",
            "account_creation_days",
            "contrib_perc_commit",
            "team_size",
            "open_issue_num",
            "open_pr_num",
            "pr_succ_rate",
            "stars",
            "description_length",
            "lifetime_minutes",
            "ci_exists",
            "num_code_comments",
            "ci_failed_perc",
            "num_commits",
            "src_churn",
            "files_added",
            "files_deleted",
            "reopen_or_not",
            "commits_on_files_touched",
            "has_comments",
            "num_comments",
            "other_comment",
        ]
    ]
    df.to_csv("data/filteredData_selected_pso.csv", index=False)


if __name__ == "__main__":
    filter_data()
    generate_data_pso()
