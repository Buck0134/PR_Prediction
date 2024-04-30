# PR_Prediction SOME CHANGE

he PR Prediction Project utilizes cutting-edge machine learning techniques to predict the likelihood of your pull request (PR) being approved. By analyzing historical data from your project's repository, including factors such as code complexity, changes made, and past PR outcomes, our model can forecast the success of new pull requests.

## How It Works

Our machine learning model has been trained on a vast dataset of pull request outcomes across various projects. It considers multiple features from each PR, including:

Developer Characteristics
Code additions and deletions
Number of files changed
Past contributor activity
Commit message contents
Current Continous Integration Set up
Once a new pull request is made, our system automatically evaluates these factors and predicts whether the PR will be accepted or needs further revisions. This not only helps streamline the review process but also guides contributors in improving their submissions for a higher chance of approval.

For more detailed information on setting up and using this feature in your projects, refer to the following information.

## To start up virtual environment and install all dependencies

    sh start.sh
    source myenv/bin/activate

## Run the following commands for data preprocessing

    python3 process/process.py

## Service is hosted separately on service folder


