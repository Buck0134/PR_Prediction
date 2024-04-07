import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

file_path = "../data/processedDataNew.csv"
df = pd.read_csv(file_path)

print(df.columns)

# X = df.drop('merged_or_not', axis=1)
# y = df['merged_or_not']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# print(y)

# mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, activation='relu', solver='adam', random_state=1)

# mlp.fit(X_train_scaled, y_train)

# # Predictions
# y_pred = mlp.predict(X_test_scaled)

# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy}')

# f1 = f1_score(y_test, y_pred)
# print(f'F1: {f1}')