from flask import Flask, request, jsonify, render_template
import requests  # Import the requests library
from datetime import datetime
import time
import logging
import numpy as np
from joblib import load
app = Flask(__name__)

# kBest:
# ['first_response_time', : V
# 'account_creation_days', : W
# 'contrib_perc_commit', : F
# 'team_size', : H
# 'open_issue_num', : I
# 'project_age', : U
# 'open_pr_num', : X
# 'pr_succ_rate', : Y
# 'stars', : K
# 'description_length',  : Z
# 'lifetime_minutes', : A1 : P
# 'ci_latency', B1
# 'ci_failed_perc', : O
# 'commits_on_files_touched', : C1
# 'num_comments', : D1
# 'merged_or_not'] : Independent


@app.route('/')
def home():
    return render_template('ci_setup_guide.html')

# health check
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "UP"}), 200

@app.route('/github/PRdata', methods=['POST'])
def github_PR_data():
    # Parse data from the request
    data = request.json
    github_token = data.get('token')
    owner = data.get('owner')
    repo = data.get('repo')
    pr_number = data.get('pr_number')
    
    # For now, just print the data or do some basic processing
    print("Received GitHub Data:", github_token, owner, repo, pr_number)
    
        # Use the GitHub API to fetch PR details
    github_api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    headers = {"Authorization": f"token {github_token}"}
    response = requests.get(github_api_url, headers=headers)
    
    if response.status_code == 200:
        pr_details = response.json()
        # Here, you can process the PR details as needed
        print("PR Title:", pr_details['title'])  # Example of using the PR data
        return jsonify({"message": "PR details fetched successfully", "PR Details": pr_details}), 200
    else:
        # Attempt to parse error details from the response
        try:
            error_details = response.json()  # Assuming the error details are in JSON format
            error_message = error_details.get('message', 'No error message provided.')
        except ValueError:
            # If the response isn't in JSON format or parsing failed
            error_message = response.text  # Fallback to the raw response text

        return jsonify({
            "message": "Failed to fetch PR details",
            "error": error_message,
            "status_code": response.status_code
        }), response.status_code
    

@app.route('/github/user', methods=['POST'])
def github_user():
    data = request.json
    github_token = data.get('token')
    username = data.get('username')
    
    if not github_token or not username:
        return jsonify({"error": "Missing token or username"}), 400

    github_api_url = f"https://api.github.com/users/{username}"
    headers = {"Authorization": f"token {github_token}"}
    response = requests.get(github_api_url, headers=headers)

    if response.status_code == 200:
        user_details = response.json()
        return jsonify({"message": "User details fetched successfully", "User Details": user_details}), 200
    else:
        return jsonify({"error": "Failed to fetch user details", "status_code": response.status_code}), response.status_code


# Placeholder prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # This is where you will process the request and return predictions
    data = request.json
    return jsonify({"message": "This is where the prediction result will be returned", "input": data}), 200


@app.route('/github/full_data', methods=['POST'])
def github_full_data():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    # Parse data from the request
    data = request.json
    github_token = data.get('token')
    owner = data.get('owner')
    repo = data.get('repo')
    developer_username = data.get('developer_username')
    pull_number = data.get('pull_number')

    logging.info(data)

    headers = {"Authorization": f"token {github_token}"}
    base_url = f"https://api.github.com/repos/{owner}/{repo}"

    # Initialize a dictionary to hold all the fetched data
    fetched_data = {}
    finalized_data = {}

    # A. Check if this is the first PR by the developer CURRENT IMPLEMENTATION IS NOT FIRST PR IN THE REPO
    search_url = f"https://api.github.com/search/issues?q=is:pr+author:{developer_username}"
    pr_response = requests.get(search_url, headers=headers)
    if pr_response.status_code == 200:
        pr_data = pr_response.json()
        finalized_data['is_first_pr'] = pr_data.get('total_count', 0) == 1
    else:
        error_message = "Failed to determine if this is the developer's first PR"
        try:
            # Attempt to parse the error message if the response is in JSON format
            error_data = pr_response.json()
            if 'message' in error_data:
                error_message += f": {error_data['message']}"
            elif 'error' in error_data:
                error_message += f": {error_data['error']}"
        except ValueError:
            # Fallback if the response is not in JSON format or .json() parsing fails
            error_message += f". Response content: {pr_response.text[:100]}"  # Show first 100 characters of the response text as an example
        finalized_data['is_first_pr'] = {"error": error_message}

    # B. Count the number of reviews made by the developer in this repository
    # Search Index Delay: GitHub's search results are based on a search index. According to GitHub's documentation, this index might not be immediately up-to-date, leading to a delay in reflecting recent activities like reviews.
    # Minutes to Hours
    review_count = get_user_review_count(github_token, developer_username)
    if review_count != -1:
        finalized_data['review_count'] = review_count
    else:
        finalized_data['review_count'] = {"error": "Failed to count reviews"}

    # ? C. core_member: paper does not specify how core_member is calculated

    # ? D. contrib/inte_X : contributor/integrator personality traits (open: openness; cons: conscientious; extra: extraver- sion; agree: agreeableness; neur: neuroticism)
    # contrib cons and contrib agree and contrib neur
    # inte open', 'inte cons', 'inte agree', 'inte neur'

    # E. prev_pullregs
    search_url = f"https://api.github.com/search/issues?q=is:pr+author:{developer_username}"
    headers = {"Authorization": f"token {github_token}"}

    response = requests.get(search_url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        finalized_data['prev_pullregs'] = data.get('total_count', 0)
    else:
        print(f"Failed to fetch user pull requests, status code: {response.status_code}")
        finalized_data['prev_pullregs'] = -1

    # F. contrib_perc_commit: % of previous contributor’s commit: I dont understand
    url = f"https://api.github.com/repos/{owner}/{repo}/stats/contributors"
    headers = {"Authorization": f"token {github_token}"}
    response = fetch_with_retry(url, headers)
    if response == None:
        pass
    elif response.status_code == 200:
        data = response.json()
        total_commits = 0
        user_commits = 0

        for contributor in data:
            total_commits += contributor['total']
            if contributor['author']['login'] == developer_username:
                user_commits = contributor['total']
        
        if total_commits > 0:
            finalized_data['contrib_perc_commit'] = (user_commits / total_commits)
        else:
            finalized_data['contrib_perc_commit'] = 0
    else:
        print(f"Failed to fetch repository statistics, status code: {response.status_code}" )
        finalized_data['contrib_perc_commit'] = -1
    
    # ? G. SLOC: Source Lines of Code
    # Needs to fetch during github CI
        # name: Count SLOC
        # on: [push]

        # jobs:
        #   sloc:
        #     runs-on: ubuntu-latest

        #     steps:
        #     - uses: actions/checkout@v2
        #     - name: Install cloc
        #       run: sudo apt-get install cloc
        #     - name: Count lines of code
        #       run: cloc . --json --out=sloc.json
        #     - name: Upload SLOC data
        #       uses: actions/upload-artifact@v2
        #       with:
        #         name: sloc
        #         path: sloc.json
    
    # H. team_size
    url = f"https://api.github.com/repos/{owner}/{repo}/contributors"
    headers = {"Authorization": f"token {github_token}"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        contributors = response.json()
        finalized_data['team_size'] = len(contributors)
    else:
        print(f"Failed to fetch contributors, status code: {response.status_code}")
        finalized_data['team_size'] = -1
    
    # I. open_issue_num
    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {"Authorization": f"token {github_token}"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        repo_info = response.json()
        finalized_data['open_issue_num'] = repo_info.get('open_issues_count', 0)
    else:
        print(f"Failed to fetch repository information, status code: {response.status_code}")
        finalized_data['open_issue_num'] = -1
    
    # ? J. test_lines_per_kloc: Github CI but hard to get

    # K. stars
    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {"Authorization": f"token {github_token}"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        repo_info = response.json()
        finalized_data['stars'] = repo_info.get('stargazers_count', 0)
    else:
        print(f"Failed to fetch repository information, status code: {response.status_code}")
        finalized_data['stars'] = -1
    
    # ? L. integrator_availability: latest activity of the two most active integrators
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    headers = {"Authorization": f"token {github_token}"}
    params = {"per_page": 1}  # We only need the most recent commit

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        commits = response.json()
        if commits:  # Check if there's at least one commit
            latest_commit_date = commits[0]['commit']['committer']['date']
            finalized_data['latest_activity'] = latest_commit_date
        else:
            finalized_data['latest_activity'] = 'No commits found'
    else:
        print(f"Failed to fetch repository commits, status code: {response.status_code}")
        finalized_data['latest_activity'] = -1 # "latest_activity": "2024-04-05T01:58:57Z"

    # ? M. test_inclusion
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/files"
    headers = {"Authorization": f"token {github_token}"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        files = response.json()
        for file in files:
            if "test" in file['filename'] or "spec" in file['filename']:
                finalized_data['test_inclusion'] = 'yes'
            else:
                finalized_data['test_inclusion'] = 'no'
    else:
        print(f"Failed to fetch PR files, status code: {response.status_code}")
        finalized_data['test_inclusion'] = 'error'
    
    # ? N. num_code_comments
    # def count_pr_comment_lines(github_token, owner, repo, pull_number):
    #     # Step 1: List files changed in the PR
    #     url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/files"
    #     headers = {"Authorization": f"token {github_token}"}
    #     response = requests.get(url, headers=headers)
    #     files = response.json()
        
    #     total_comment_lines = 0
    #     for file in files:
    #         # Step 2: Fetch file contents (simplified; actual implementation may vary)
    #         file_url = file['contents_url']
    #         file_response = requests.get(file_url, headers=headers)
    #         file_contents = file_response.text  # Simplification; actual content fetching is more complex
            
    #         # Step 3: Count comment lines (highly simplified; would need language-specific parsing)
    #         # This is a placeholder for where you would parse the file contents
    #         # and count comment lines based on the file's programming language.
    #         comment_lines = count_comments_for_language(file_contents, file['filename'])
    #         total_comment_lines += comment_lines
        
    #     return total_comment_lines

    # def count_comments_for_language(contents, filename):
    #     # Placeholder function for counting comment lines
    #     # This would need to be implemented based on the programming language
    #     # and could get quite complex.
    #     return 0

    # * O. ci_failed_perc: what should my value be if there is no CI
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs"
    headers = {"Authorization": f"token {github_token}"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        runs = response.json()['workflow_runs']
        total_runs = len(runs)
        failed_runs = sum(1 for run in runs if run['conclusion'] == 'failure')
        
        if total_runs > 0:
            failed_percentage = (failed_runs / total_runs) * 100
            finalized_data['ci_failed_perc'] = failed_percentage
        else:
            finalized_data['ci_failed_perc'] = 0  # No runs found
    else:
        print(f"Failed to fetch workflow runs, status code: {response.status_code}")
        finalized_data['ci_failed_perc'] = -1
    
    # P. files_changed, files_added, files_deleted
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/files"
    headers = {"Authorization": f"token {github_token}"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        files = response.json()
        changes = {
            'added': sum(1 for f in files if f['status'] == 'added'),
            'modified': sum(1 for f in files if f['status'] == 'modified'),
            'deleted': sum(1 for f in files if f['status'] == 'deleted')
        }
        finalized_data['files_added'] = changes['added']
        finalized_data['files_changed'] = changes['modified']
        finalized_data['files_deleted'] = changes['deleted']
    else:
        print(f"Failed to fetch PR files, status code: {response.status_code}")
        finalized_data['files_added'] = -1
        finalized_data['files_changed'] = -1
        finalized_data['files_deleted'] = -1
    
    # Q. friday_effect
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}"
    headers = {"Authorization": f"token {github_token}"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        pr_details = response.json()
        created_at = pr_details['created_at']
        created_date = datetime.strptime(created_at, '%Y-%m-%dT%H:%M:%SZ')
        
        # datetime.weekday() returns 0 for Monday and 6 for Sunday
        if created_date.weekday() == 4:  # Friday
            finalized_data['friday_effect'] = 'yes'
        else:
            finalized_data['friday_effect'] = 'no'
    else:
        print(f"Failed to fetch PR details, status code: {response.status_code}")
        finalized_data['friday_effect'] = 'error'
    
    # R. reopen_or_not
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pull_number}/events"
    headers = {"Authorization": f"token {github_token}"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        events = response.json()
        for event in events:
            if event['event'] == 'reopened':
                finalized_data['reopen_or_not'] = 'yes'
            else:
                finalized_data['reopen_or_not'] = 'no'
    else:
        print(f"Failed to fetch PR events, status code: {response.status_code}")
        finalized_data['reopen_or_not'] = 'error'
    
    # S. has_comments
    headers = {"Authorization": f"token {github_token}"}
    # Check for review comments
    review_comments_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/comments"
    review_comments_response = requests.get(review_comments_url, headers=headers)
    # Check for issue comments
    issue_comments_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pull_number}/comments"
    issue_comments_response = requests.get(issue_comments_url, headers=headers)
    if review_comments_response.status_code == 200 and issue_comments_response.status_code == 200:
        review_comments = review_comments_response.json()
        issue_comments = issue_comments_response.json()
        if review_comments or issue_comments:
            finalized_data['has_comments'] = 'yes'
        else:
            finalized_data['has_comments'] = 'no'
    else:
        print("Failed to fetch PR comments")
        finalized_data['has_comments'] = 'error'
    
    # # T. other_comment
    # commenters = set()

    # # Function to perform GET requests and check status code
    # def get_json(url):
    #     response = requests.get(url, headers=headers)
    #     if response.status_code == 200:
    #         return response.json()
    #     else:
    #         print(f"Failed to fetch data from {url}, status code: {response.status_code}")
    #         return None

    # # Fetch PR issue comments
    # issue_comments_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pull_number}/comments"
    # issue_comments = get_json(issue_comments_url)
    # if issue_comments is None:
    #     return 'error: failed to fetch issue comments'
    # for comment in issue_comments:
    #     commenters.add(comment['user']['login'])

    # # Fetch PR review comments
    # review_comments_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/comments"
    # review_comments = get_json(review_comments_url)
    # if review_comments is None:
    #     return 'error: failed to fetch review comments'
    # for comment in review_comments:
    #     commenters.add(comment['user']['login'])

    # # Fetch repository contributors
    # contributors_url = f"https://api.github.com/repos/{owner}/{repo}/contributors"
    # contributors = get_json(contributors_url)
    # if contributors is None:
    #     return 'error: failed to fetch contributors'
    # contributor_logins = {contributor['login'] for contributor in contributors}

    # # Check for non-contributor comments
    # non_contributor_comments = commenters - contributor_logins

    # finalized_data['other_comment'] = 'yes' if non_contributor_comments else 'no'
    
    # Starting GA: 
    # U. project_age
    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {"Authorization": f"token {github_token}"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        repo_details = response.json()
        created_at = repo_details['created_at']
        created_date = datetime.strptime(created_at, '%Y-%m-%dT%H:%M:%SZ')
        current_date = datetime.now()
        age_months = (current_date - created_date).days / 31
        finalized_data['project_age'] = int(age_months)
    else:
        print(f"Failed to fetch repository details, status code: {response.status_code}")
        finalized_data['project_age'] = "error"

    # V. first_response_time
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    pr_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}"
    timeline_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pull_number}/timeline"

    pr_response = requests.get(pr_url, headers=headers)
    if pr_response.status_code != 200:
        finalized_data['first_response_time'] = None
    
    pr_created_at = datetime.strptime(pr_response.json()['created_at'], '%Y-%m-%dT%H:%M:%SZ')

    timeline_response = requests.get(timeline_url, headers=headers)
    if timeline_response.status_code != 200:
        finalized_data['first_response_time'] = None
    first_response_time = None
    for event in timeline_response.json():
        if event['event'] in ['commented', 'reviewed']:  # Adjust as necessary
            event_time = datetime.strptime(event['created_at'], '%Y-%m-%dT%H:%M:%SZ')
            if not first_response_time or event_time < first_response_time:
                first_response_time = event_time
    
    if first_response_time:
        delta = first_response_time - pr_created_at
        finalized_data['first_response_time'] = delta.total_seconds() / 60
    else:
        finalized_data['first_response_time'] = None
    
    # W. account_creation_days
    url = f"https://api.github.com/users/{developer_username}"
    response = requests.get(url)
    
    if response.status_code == 200:
        user_info = response.json()
        created_at = user_info['created_at']
        creation_date = datetime.strptime(created_at, '%Y-%m-%dT%H:%M:%SZ')
        current_date = datetime.now()
        account_age_days = (current_date - creation_date).days
        finalized_data['account_creation_days'] = account_age_days
    else:
        finalized_data['account_creation_days'] = f"Failed to fetch user information, status code: {response.status_code}"

    # X. open_pr_num
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls?state=open"
    headers = {}
    headers['Authorization'] = f'token {github_token}'
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        open_prs = response.json()
        finalized_data['open_pr_num'] = len(open_prs)  # Returns the count of open PRs
    else:
        finalized_data['open_pr_num'] = -1
    
    # Y. pr_succ_rate
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls?state=closed"
    headers = {"Authorization": f"token {github_token}"} if github_token else {}
    prs = []
    while url:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            prs.extend(response.json())
            url = response.links.get('next', {}).get('url', None)  # Handle pagination
        else:
            return f"Failed to fetch PRs, status code: {response.status_code}"

    merged_count = sum(1 for pr in prs if pr['merged_at'])
    total_closed_count = len(prs)

    if total_closed_count > 0:
        success_rate = (merged_count / total_closed_count)
        finalized_data['pr_succ_rate'] =  success_rate
    else:
        finalized_data['pr_succ_rate'] =  None

    # Z. description_length
    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {"Authorization": f"token {github_token}"} if github_token else {}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        repo_details = response.json()
        description = repo_details.get('description', '')
        if description:  # Check if the description is not None or empty
            finalized_data['description_length'] = len(description)
        else:
            finalized_data['description_length'] =  0  # Return 0 if there is no description
    else:
        print(f"Failed to fetch repository details, status code: {response.status_code}")
        finalized_data['description_length'] =  None

    # A1. lifetime_minutes
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}"
    headers = {"Authorization": f"token {github_token}"} if github_token else {}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        pr_details = response.json()
        created_at = datetime.strptime(pr_details['created_at'], '%Y-%m-%dT%H:%M:%SZ')
        
        if pr_details['closed_at']:
            closed_at = datetime.strptime(pr_details['closed_at'], '%Y-%m-%dT%H:%M:%SZ')
            lifetime_minutes = (closed_at - created_at).total_seconds() / 60
            finalized_data['lifetime_minutes'] = int(lifetime_minutes)
        else:
            finalized_data['lifetime_minutes'] = 0
    else:
        finalized_data['lifetime_minutes'] = -1

    # B1. ci_latency
    pr_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}"
    runs_url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs?event=pull_request&pull_request_number={pull_number}"

    # Fetch PR creation time
    pr_response = requests.get(pr_url, headers=headers)
    if pr_response.status_code != 200:
        return "Failed to fetch PR details."
    pr_created_at = datetime.strptime(pr_response.json()['created_at'], '%Y-%m-%dT%H:%M:%SZ')

    # Fetch workflow runs
    runs_response = requests.get(runs_url, headers=headers)
    if runs_response.status_code != 200:
        return "Failed to fetch workflow runs."
    
    runs = runs_response.json()['workflow_runs']
    completed_runs = [run for run in runs if run['status'] == 'completed']
    if not completed_runs:
        return "No completed CI runs found."

    # Find the earliest completed CI run
    first_run_finished = min(completed_runs, key=lambda run: run['updated_at'])
    first_run_finished_at = datetime.strptime(first_run_finished['updated_at'], '%Y-%m-%dT%H:%M:%SZ')

    # Calculate latency
    ci_latency = (first_run_finished_at - pr_created_at).total_seconds() / 60

    finalized_data['ci_latency'] = ci_latency

    # C1. commits_on_files_touched
    pr_files_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/files"
    files_response = requests.get(pr_files_url, headers=headers)
    
    if files_response.status_code != 200:
        return "Failed to fetch PR files."
    
    total_commits = 0
    for file in files_response.json():
        commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits?path={file['filename']}"
        commits_response = requests.get(commits_url, headers=headers)
        if commits_response.status_code == 200:
            # Each commit in the history of the file touched by the PR
            total_commits += len(commits_response.json())
        else:
            print(f"Failed to fetch commits for file: {file['filename']}")
    
    finalized_data['commits_on_files_touched'] = total_commits

    # D1. num_comments
    # Fetch issue comments (general PR comments)
    issue_comments_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pull_number}/comments"
    issue_comments_response = requests.get(issue_comments_url, headers=headers)
    if issue_comments_response.status_code != 200:
        return "Failed to fetch issue comments."
    issue_comments_count = len(issue_comments_response.json())
    
    # Fetch review comments (code comments)
    review_comments_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/comments"
    review_comments_response = requests.get(review_comments_url, headers=headers)
    if review_comments_response.status_code != 200:
        return "Failed to fetch review comments."
    review_comments_count = len(review_comments_response.json())
    
    # Total comments
    total_comments = issue_comments_count + review_comments_count
    finalized_data['num_comments'] = total_comments

    print(finalized_data)
    kbest = ['first_response_time', 'account_creation_days', 'contrib_perc_commit', 'team_size', 'open_issue_num', 'project_age', 'open_pr_num', 'pr_succ_rate', 'stars', 'description_length', 'lifetime_minutes', 'ci_latency', 'ci_failed_perc', 'commits_on_files_touched', 'num_comments']
    passingInResult = {}
    for key in finalized_data:
        if key in kbest:
            passingInResult[key] = finalized_data[key]
    
    print(passingInResult)
    model = load('../random_forest_kBest.joblib')
    modelUsed = "random_forest_kBest"
    for key, value in passingInResult.items():
        if value is None:
            passingInResult[key] = np.nan

    # Convert to a 2D array (1 row, many columns)
    X = np.array([list(passingInResult.values())])
    predictionResult = model.predict(X)

    print(predictionResult)
    if predictionResult == 0:
        return jsonify(f"We are using {modelUsed} model \n" + "We estimate this PR will be DENIED" )
    else:
        return jsonify(f"We are using {modelUsed} model \n" + "We estimate this PR will be PASSED")

def get_user_review_count(github_token, developer_username):
    """
    Fetches the number of pull request reviews made by a specified user on GitHub.

    Parameters:
    - github_token: str - A GitHub personal access token for authentication.
    - developer_username: str - The GitHub username of the developer.

    Returns:
    - int: The count of pull request reviews made by the user.
    """

    # GitHub GraphQL API URL
    graphql_url = 'https://api.github.com/graphql'

    # Simplified GraphQL query to count reviewed pull requests
    query = """
    query UserReviewCount($username: String!) {
      search(query: "type:pr reviewed-by:" + $username, type: ISSUE, first: 1) {
        issueCount
      }
    }
    """

    # JSON payload for the request
    json_payload = {
        "query": query,
        "variables": {"username": developer_username}
    }

    # Request headers
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Content-Type": "application/json",
    }

    # Make the request to the GitHub GraphQL API
    response = requests.post(graphql_url, json=json_payload, headers=headers)
    # print(response)
    # Check if the request was successful
    if response.status_code == 200:
        response_data = response.json()
        review_count = response_data.get("data", {}).get("search", {}).get("issueCount", 0)
        return review_count
    else:
        print(f"Failed to fetch user reviews, status code: {response.status_code}")
        return -1  # Indicates an error occurred

def get_user_pr_count(github_token, developer_username):
    """
    Fetches the number of pull requests authored by a specified user on GitHub.

    Parameters:
    - github_token: str - A GitHub personal access token for authentication.
    - developer_username: str - The GitHub username of the developer.

    Returns:
    - int: The count of pull requests authored by the user.
    """

    # GitHub GraphQL API URL
    graphql_url = 'https://api.github.com/graphql'

    # GraphQL query
    query = """
    query UserPullRequestCount($username: String!) {
      search(query: "type:pr author:" + $username, type: ISSUE, first: 1) {
        issueCount
      }
    }
    """

    # JSON payload for the request
    json_payload = {
        "query": query,
        "variables": {"username": developer_username}
    }

    # Request headers
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Content-Type": "application/json",
    }

    # Make the request to the GitHub GraphQL API
    response = requests.post(graphql_url, json=json_payload, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        response_data = response.json()
        pr_count = response_data.get("data", {}).get("search", {}).get("issueCount", 0)
        return pr_count
    else:
        print(f"Failed to fetch user pull requests, status code: {response.status_code}")
        return -1  # Indicates an error occurred
    
def fetch_with_retry(url, headers, max_attempts=10):
    for attempt in range(max_attempts):
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response  # Success
        elif response.status_code == 202:
            print("Data is being prepared, retrying...")
            time.sleep(5)  # Wait for 5 seconds before retrying
        else:
            # Handle other HTTP errors
            return response
    return None  # Exceeded max attempts


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
