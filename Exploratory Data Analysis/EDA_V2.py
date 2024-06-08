import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('heart.csv')

# Define the features and the target
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
target = 'target'

X = df[features]
y = df[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

# Evaluate Logistic Regression Model
y_pred_logistic = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(y_test, y_pred_logistic)
logistic_precision = precision_score(y_test, y_pred_logistic)
logistic_recall = recall_score(y_test, y_pred_logistic)
logistic_f1 = f1_score(y_test, y_pred_logistic)

print(f'Logistic Regression Accuracy: {logistic_accuracy}')
print(f'Logistic Regression Precision: {logistic_precision}')
print(f'Logistic Recall Score: {logistic_recall}')
print(f'Logistic Regression F1 Score: {logistic_f1}')

# Save the Logistic Regression model
joblib.dump(logistic_model, 'Exploratory Data Analysis\\logistic_regression_model.pkl')

# Train Bagged Decision Tree Model
bagging_model = BaggingClassifier(DecisionTreeClassifier(), n_estimators=50, random_state=42)
bagging_model.fit(X_train, y_train)

# Evaluate Bagged Decision Tree Model
y_pred_bagging = bagging_model.predict(X_test)
bagging_accuracy = accuracy_score(y_test, y_pred_bagging)
print(f'Bagged Decision Tree Accuracy: {bagging_accuracy}')

# Save the Bagged Decision Tree model
joblib.dump(bagging_model, 'bagged_decision_tree_model.pkl')

# Compare the models
if logistic_accuracy > bagging_accuracy:
    print("Logistic Regression is the better model.")
elif logistic_accuracy < bagging_accuracy:
    print("Bagged Decision Tree is the better model.")
else:
    print("Both models have the same accuracy.")

# Plot the comparison
models = ['Logistic Regression', 'Bagged Decision Tree']
accuracies = [logistic_accuracy, bagging_accuracy]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'green'])
plt.ylim([0, 1])
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.show()

# Outlier detection and removal using IQR
# ============================================
print("\n================ Outlier Detection and Removal ==================")

def remove_outliers(df, features):
    initial_rows = df.shape[0]
    outliers_removed = 0

    for feature in features:
        intitial_feature_rows = df.shape[0]

        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]

        feature_outliers_removed = intitial_feature_rows - df.shape[0]
        outliers_removed += feature_outliers_removed

    print(f'Total outliers removed: {initial_rows - df.shape[0]}')
    return df

df_cleaned = remove_outliers(df, features)

# Define features and target
X_cleaned = df_cleaned[features]
y_cleaned = df_cleaned[target]

# Split cleaned data
X_train_cleaned, X_test_cleaned, y_train_cleaned, y_test_cleaned = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

# Train and evaluate Bagged Decision Tree model (Cleaned)
bagging_model_cleaned = BaggingClassifier(DecisionTreeClassifier(), n_estimators=50, random_state=42)
bagging_model_cleaned.fit(X_train_cleaned, y_train_cleaned)
y_pred_bagging_cleaned = bagging_model_cleaned.predict(X_test_cleaned)

bagging_accuracy_cleaned = accuracy_score(y_test_cleaned, y_pred_bagging_cleaned)
bagging_precision_cleaned = precision_score(y_test_cleaned, y_pred_bagging_cleaned)
bagging_recall_cleaned = recall_score(y_test_cleaned, y_pred_bagging_cleaned)
bagging_f1_cleaned = f1_score(y_test_cleaned, y_pred_bagging_cleaned)

print("Bagged Decision Tree Metrics (Cleaned): ")
print(f'Accuracy: {bagging_accuracy_cleaned}')
print(f'Precision: {bagging_precision_cleaned}')
print(f'Recall: {bagging_recall_cleaned}')
print(f'F1 Score: {bagging_f1_cleaned}')
print('\n')

# Train and evaluate Rando model (Cleaned)
logistic_model_cleaned = LogisticRegression(max_iter=1000)
logistic_model_cleaned.fit(X_train_cleaned, y_train_cleaned)
y_pred_logistic_cleaned = logistic_model_cleaned.predict(X_test_cleaned)

logistic_accuracy_cleaned = accuracy_score(y_test_cleaned, y_pred_logistic_cleaned)
logistic_precision_cleaned = precision_score(y_test_cleaned, y_pred_logistic_cleaned)
logistic_recall_cleaned = recall_score(y_test_cleaned, y_pred_logistic_cleaned)
logistic_f1_cleaned = f1_score(y_test_cleaned, y_pred_logistic_cleaned)

print("Logistic Regression Metrics (Cleaned): ")
print(f'Accuracy: {logistic_accuracy_cleaned}')
print(f'Precision: {logistic_precision_cleaned}')
print(f'Recall: {logistic_recall_cleaned}')
print(f'F1 Score: {logistic_f1_cleaned}')
print('\n')

# Labels for confusion matrices
labels = ['Does Not Have Heart Disease', 'Has Heart Disease']

# Plot confusion matrix for Logistic Regression
logistic_confusion_matrix = confusion_matrix(y_test_cleaned, y_pred_logistic_cleaned)
plt.figure(figsize=(10, 6))
sns.heatmap(logistic_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

# Plot confusion matrix for Bagged Descision Tree
bagged_confusion_matrix = confusion_matrix(y_test_cleaned, y_pred_bagging_cleaned)
plt.figure(figsize=(10, 6))
sns.heatmap(bagged_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Bagged Decision Tree')
plt.show()