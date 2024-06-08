import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# Load the dataset
heart_data = pd.read_csv('heart.csv')

# Prepare the data
X = heart_data.drop(columns='target')
y = heart_data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)
y_prob_log_reg = log_reg.predict_proba(X_test_scaled)[:, 1]

# Bagged Decision Tree model
bagged_tree = BaggingClassifier(DecisionTreeClassifier(), random_state=42)
bagged_tree.fit(X_train_scaled, y_train)
y_pred_bagged_tree = bagged_tree.predict(X_test_scaled)
y_prob_bagged_tree = bagged_tree.predict_proba(X_test_scaled)[:, 1]

# Evaluation metrics
f1_log_reg = f1_score(y_test, y_pred_log_reg)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
f1_bagged_tree = f1_score(y_test, y_pred_bagged_tree)
accuracy_bagged_tree = accuracy_score(y_test, y_pred_bagged_tree)

# ROC curve
fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, y_prob_log_reg)
fpr_bagged_tree, tpr_bagged_tree, _ = roc_curve(y_test, y_prob_bagged_tree)
roc_auc_log_reg = roc_auc_score(y_test, y_prob_log_reg)
roc_auc_bagged_tree = roc_auc_score(y_test, y_prob_bagged_tree)

# Confusion Matrix Display for both models
ConfusionMatrixDisplay.from_estimator(log_reg, X_test_scaled, y_test)
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

ConfusionMatrixDisplay.from_estimator(bagged_tree, X_test_scaled, y_test)
plt.title('Confusion Matrix - Bagged Decision Tree')
plt.show()

# Evaluation metrics for Logistic Regression
log_reg_report = classification_report(y_test, y_pred_log_reg, output_dict=True)
log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
log_reg_precision = log_reg_report['1']['precision']
log_reg_recall = log_reg_report['1']['recall']
log_reg_f1 = log_reg_report['1']['f1-score']

# Evaluation metrics for Bagged Decision Tree
bagged_tree_report = classification_report(y_test, y_pred_bagged_tree, output_dict=True)
bagged_tree_accuracy = accuracy_score(y_test, y_pred_bagged_tree)
bagged_tree_precision = bagged_tree_report['1']['precision']
bagged_tree_recall = bagged_tree_report['1']['recall']
bagged_tree_f1 = bagged_tree_report['1']['f1-score']

print("Logistic Regression Metrics: ")
print(f'Accuracy: {log_reg_accuracy}')
print(f'Precision: {log_reg_precision}')
print(f'Recall: {log_reg_recall}')
print(f'F1 Score: {log_reg_f1}')

print("\nBagged Decision Tree Metrics: ")
print(f'Accuracy: {bagged_tree_accuracy}')
print(f'Precision: {bagged_tree_precision}')
print(f'Recall: {bagged_tree_recall}')
print(f'F1 Score: {bagged_tree_f1}')

# SAVE OUR MODELS TO FILE!!!!
joblib.dump(bagged_tree, 'bagged_decision_tree_model.pkl')
joblib.dump(log_reg, 'logistic_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler as well

# Input values that should ideally result in a healthy prediction (0)
# 58,0,0,100,248,0,0,122,0,1,1,0