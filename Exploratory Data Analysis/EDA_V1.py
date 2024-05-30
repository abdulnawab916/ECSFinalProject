# THIS SHOWED US THAT BAGGED DECISION TREE AND RANDOM FOREST HAD THE SAME ACCURACY.
# Bagged Decision Tree Accuracy: 0.9853658536585366
# Random Forest Accuracy: 0.9853658536585366
# Bagged Decision Tree Cross-Validation Accuracy: 0.9970731707317073
# Random Forest Cross-Validation Accuracy: 0.9970731707317073

# BAGGED DECISION TREE
# ============================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_csv('Exploratory Data Analysis/heart.csv')

# Define the features and the target
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
target = 'target'

X = df[features]
y = df[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the bagged decision tree model
bagging_model = BaggingClassifier(DecisionTreeClassifier(), n_estimators=50, random_state=42)
bagging_model.fit(X_train, y_train)

# Evaluate the model
y_pred_bagging = bagging_model.predict(X_test)
bagging_accuracy = accuracy_score(y_test, y_pred_bagging)
print(f'Bagged Decision Tree Accuracy: {bagging_accuracy}')

# Save the model
joblib.dump(bagging_model, 'Exploratory Data Analysis\\bagged_decision_tree_model.pkl')


# RANDOM FOREST
# ============================================

from sklearn.ensemble import RandomForestClassifier

# Train the random forest model
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

# Evaluate the model
y_pred_rf = random_forest_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {rf_accuracy}')

# Save the model
joblib.dump(random_forest_model, 'Exploratory Data Analysis\\random_forest_model.pkl')


# COMPARE THE MODELS
# =============================================

# Define the features and the target
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
target = 'target'

X = df[features]
y = df[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate Bagged Decision Tree
bagging_model = BaggingClassifier(DecisionTreeClassifier(), n_estimators=50, random_state=42)
bagging_model.fit(X_train, y_train)
y_pred_bagging = bagging_model.predict(X_test)
bagging_accuracy = accuracy_score(y_test, y_pred_bagging)

# Train and evaluate Random Forest
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)
y_pred_rf = random_forest_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print(f'Bagged Decision Tree Accuracy: {bagging_accuracy}')
print(f'Random Forest Accuracy: {rf_accuracy}')


# CROSS-VALIDATION
# =================================================

from sklearn.model_selection import cross_val_score

# Cross-validation for Bagged Decision Tree
bagging_cv_scores = cross_val_score(bagging_model, X, y, cv=5, scoring='accuracy')
print(f'Bagged Decision Tree Cross-Validation Accuracy: {bagging_cv_scores.mean()}')

# Cross-validation for Random Forest
rf_cv_scores = cross_val_score(random_forest_model, X, y, cv=5, scoring='accuracy')
print(f'Random Forest Cross-Validation Accuracy: {rf_cv_scores.mean()}')


# FEATURE IMPORTANCE
# ==================================================

import matplotlib.pyplot as plt

# Get feature importances
feature_importances = random_forest_model.feature_importances_

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances in Random Forest')
plt.gca().invert_yaxis()
plt.show()
