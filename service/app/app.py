from flask import Flask, request, jsonify
import requests  # Import the requests library
from datetime import datetime
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

    headers = {"Authorization": f"token {github_token}"}
    base_url = f"https://api.github.com/repos/{owner}/{repo}"

    # Initialize a dictionary to hold all the fetched data
    fetched_data = {}

    # API Calls and storing their whole responses
    # 1. Fetch Repository Details
    repo_response = requests.get(base_url, headers=headers)
    fetched_data['repo_details'] = repo_response.json() if repo_response.status_code == 200 else {"error": "Failed to fetch repository details"}

    # 2. Open Pull Requests
    open_prs_response = requests.get(f"{base_url}/pulls?state=open", headers=headers)
    fetched_data['open_prs'] = open_prs_response.json() if open_prs_response.status_code == 200 else {"error": "Failed to fetch open PRs"}

    # 3. Open Issues
    open_issues_response = requests.get(f"{base_url}/issues?state=open", headers=headers)
    fetched_data['open_issues'] = open_issues_response.json() if open_issues_response.status_code == 200 else {"error": "Failed to fetch open issues"}

    # 4. Contributors and Team Size
    contributors_response = requests.get(f"{base_url}/contributors", headers=headers)
    fetched_data['contributors'] = contributors_response.json() if contributors_response.status_code == 200 else {"error": "Failed to fetch contributors"}

    # 5. Closed Pull Requests (for calculating PR success rate and first PR merged or not)
    closed_prs_response = requests.get(f"{base_url}/pulls?state=closed", headers=headers)
    fetched_data['closed_prs'] = closed_prs_response.json() if closed_prs_response.status_code == 200 else {"error": "Failed to fetch closed PRs"}

    # 6. Workflow Runs (for CI/CD metrics)
    workflow_runs_response = requests.get(f"{base_url}/actions/runs", headers=headers)
    fetched_data['workflow_runs'] = workflow_runs_response.json()['workflow_runs'] if workflow_runs_response.status_code == 200 else {"error": "Failed to fetch workflow runs"}

    # 7. Commits (for commit activity analysis)
    commits_response = requests.get(f"{base_url}/commits", headers=headers)
    fetched_data['commits'] = commits_response.json() if commits_response.status_code == 200 else {"error": "Failed to fetch commits"}

    # Return the fetched data as JSON
    return jsonify(fetched_data)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
