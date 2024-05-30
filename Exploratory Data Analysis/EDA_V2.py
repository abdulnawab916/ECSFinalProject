import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
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
print(f'Logistic Regression Accuracy: {logistic_accuracy}')

# Save the Logistic Regression model
joblib.dump(logistic_model, 'logistic_regression_model.pkl')

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

# Labels for confusion matrices
labels = ['Does Not Have Heart Disease', 'Has Heart Disease']

# Plot confusion matrix for Logistic Regression
logistic_confusion_matrix = confusion_matrix(y_test, y_pred_logistic)
plt.figure(figsize=(10, 6))
sns.heatmap(logistic_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

# Plot confusion matrix for Bagged Descision Tree
bagged_confusion_matrix = confusion_matrix(y_test, y_pred_bagging)
plt.figure(figsize=(10, 6))
sns.heatmap(bagged_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Bagged Decision Tree')
plt.show()