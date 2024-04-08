import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

df = pd.read_csv('../data/processedDataNew.csv')

imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Split data into features and target
X = df_imputed.drop('merged_or_not', axis=1)
y = df_imputed['merged_or_not']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(y_train)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

parameter_space = {
    'hidden_layer_sizes': [(50,), (50,100), (100,)],
    'alpha': [0.001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

mlp = MLPClassifier(max_iter=100,activation='relu', solver='adam', random_state=42)
# mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, activation='relu', solver='adam', random_state=42)
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(X_train, y_train)
print("Best parameters found:\n", clf.best_params_)

# Train the model
# mlp.fit(X_train_scaled, y_train)


# y_pred = mlp.predict(X_test_scaled)

y_pred = clf.predict(X_test_scaled)


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


f1 = f1_score(y_test, y_pred)
print(f'F1: {f1}')

precision = precision_score(y_test, y_pred)
print(f'Precision: {precision}')

recall = recall_score(y_test, y_pred)
print(f'Recall: {recall}')



