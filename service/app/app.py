from flask import Flask, request, jsonify
import requests  # Import the requests library
from datetime import datetime
import time
app = Flask(__name__)


#'prior_review_num', 'first_response_time', 'inte_open', 'account_creation_days'
#'contrib_perc_commit', 'team_size', 'open_issue_num', 'project_age', 'open_pr_num', 
#'pr_succ_rate', 'test_lines_per_kloc', 'stars', 'integrator_availability', 'description_length', 
#'lifetime_minutes', 'ci_latency', 'ci_failed_perc', 'commits_on_files_touched', 'num_comments', 
#'first_pr'
# independent: 'merged_or_not'

# Example route for health check
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
    # For now, we return a simple message
    data = request.json
    return jsonify({"message": "This is where the prediction result will be returned", "input": data}), 200


@app.route('/github/full_data', methods=['POST'])
def github_full_data():
    # Parse data from the request
    data = request.json
    github_token = data.get('token')
    owner = data.get('owner')
    repo = data.get('repo')
    developer_username = data.get('developer_username')
    pull_number = data.get('pull_number')

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
        finalized_data['is_first_pr'] = {"error": "Failed to determine if this is the developer's first PR"}

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
            finalized_data['contrib_perc_commit'] = (user_commits / total_commits) * 100
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
    
    # H. team size
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
    
    # T. other_comment
    commenters = set()

    # Function to perform GET requests and check status code
    def get_json(url):
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch data from {url}, status code: {response.status_code}")
            return None

    # Fetch PR issue comments
    issue_comments_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pull_number}/comments"
    issue_comments = get_json(issue_comments_url)
    if issue_comments is None:
        return 'error: failed to fetch issue comments'
    for comment in issue_comments:
        commenters.add(comment['user']['login'])

    # Fetch PR review comments
    review_comments_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/comments"
    review_comments = get_json(review_comments_url)
    if review_comments is None:
        return 'error: failed to fetch review comments'
    for comment in review_comments:
        commenters.add(comment['user']['login'])

    # Fetch repository contributors
    contributors_url = f"https://api.github.com/repos/{owner}/{repo}/contributors"
    contributors = get_json(contributors_url)
    if contributors is None:
        return 'error: failed to fetch contributors'
    contributor_logins = {contributor['login'] for contributor in contributors}

    # Check for non-contributor comments
    non_contributor_comments = commenters - contributor_logins

    finalized_data['other_comment'] = 'yes' if non_contributor_comments else 'no'
    
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
        age_years = (current_date - created_date).days / 365.25
        finalized_data['project_age'] = f"{age_years:.2f} years"
    else:
        print(f"Failed to fetch repository details, status code: {response.status_code}")
        finalized_data['project_age'] = "error"
    # # API Calls and storing their whole responses
    # # 1. Fetch Repository Details
    # repo_response = requests.get(base_url, headers=headers)
    # fetched_data['repo_details'] = repo_response.json() if repo_response.status_code == 200 else {"error": "Failed to fetch repository details"}

    # # 2. Open Pull Requests
    # open_prs_response = requests.get(f"{base_url}/pulls?state=open", headers=headers)
    # fetched_data['open_prs'] = open_prs_response.json() if open_prs_response.status_code == 200 else {"error": "Failed to fetch open PRs"}

    # # 3. Open Issues
    # open_issues_response = requests.get(f"{base_url}/issues?state=open", headers=headers)
    # fetched_data['open_issues'] = open_issues_response.json() if open_issues_response.status_code == 200 else {"error": "Failed to fetch open issues"}

    # # 4. Contributors and Team Size
    # contributors_response = requests.get(f"{base_url}/contributors", headers=headers)
    # fetched_data['contributors'] = contributors_response.json() if contributors_response.status_code == 200 else {"error": "Failed to fetch contributors"}

    # # 5. Closed Pull Requests (for calculating PR success rate and first PR merged or not)
    # closed_prs_response = requests.get(f"{base_url}/pulls?state=closed", headers=headers)
    # fetched_data['closed_prs'] = closed_prs_response.json() if closed_prs_response.status_code == 200 else {"error": "Failed to fetch closed PRs"}

    # # 6. Workflow Runs (for CI/CD metrics)
    # workflow_runs_response = requests.get(f"{base_url}/actions/runs", headers=headers)
    # fetched_data['workflow_runs'] = workflow_runs_response.json()['workflow_runs'] if workflow_runs_response.status_code == 200 else {"error": "Failed to fetch workflow runs"}

    # # 7. Commits (for commit activity analysis)
    # commits_response = requests.get(f"{base_url}/commits", headers=headers)
    # fetched_data['commits'] = commits_response.json() if commits_response.status_code == 200 else {"error": "Failed to fetch commits"}

    # # Return the fetched data as JSON
    print(finalized_data)
    return jsonify(finalized_data)

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
