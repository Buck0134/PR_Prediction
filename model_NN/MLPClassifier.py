import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE

df = pd.read_csv('../data/processedDataNew.csv')
smote = SMOTE(random_state=42)


# Split data into features and target
X = df.drop('merged_or_not', axis=1)
y = df['merged_or_not']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
imputer = SimpleImputer(strategy='mean')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train_imputed, y_train)




# mlp = MLPClassifier(max_iter=100,activation='relu', solver='adam', random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, activation='relu', solver='adam', random_state=42)

print(pd.Series(y_train_oversampled).value_counts())


