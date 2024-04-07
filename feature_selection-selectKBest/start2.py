import pandas as pd
from FeatureSelection_2 import PullRequestPredictor

results = []
# for N in range(30, 19, -1):  # N from 30 to 20
# Remember to mention the following error when M and N too large:
# /Users/yukewu/Desktop/SE/18-668/.venv/lib/python3.12/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
#   warnings.warn("Maximum Likelihood optimization failed to "
# /Users/yukewu/Desktop/SE/18-668/.venv/lib/python3.12/site-packages/statsmodels/regression/mixed_linear_model.py:2201: ConvergenceWarning: Retrying MixedLM optimization with lbfgs
#   warnings.warn(
# /Users/yukewu/Desktop/SE/18-668/.venv/lib/python3.12/site-packages/statsmodels/regression/mixed_linear_model.py:1635: UserWarning: Random effects covariance is singular
#   warnings.warn(msg)
# Why: When N and M are too large, the model may not converge or the random effects covariance matrix may be singular.

df_copy = pd.read_csv('../data/processedData.csv')

for N in range(15, 4, -1):  # N from 15 to 5
    print(f"N={N}, M={15 - N}")
    M = 15 - N
    
    df = df_copy.copy()
    predictor = PullRequestPredictor(df)
    
    print("Processing data...")
    predictor.preprocess()

    print("Selecting features...")
    predictor.select_features(N, M)
    
    print("Fitting and evaluating...")
    result = predictor.fit_and_evaluate()
    print(f"Result: {result}")
    results.append(result)

# Save results to CSV
result_df = pd.DataFrame(results)
result_df.to_csv('result2.csv', index=False)