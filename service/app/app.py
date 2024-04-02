from flask import Flask, request, jsonify
import requests  # Import the requests library

app = Flask(__name__)

# Example route for health check
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "UP"}), 200

@app.route('/github/PRdata', methods=['POST'])
def github_data():
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
