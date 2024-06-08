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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the dataset
df = pd.read_csv('heart.csv')

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
bagging_precision = precision_score(y_test, y_pred_bagging)
bagging_recall = precision_score(y_test, y_pred_bagging)
bagging_f1 = f1_score(y_test, y_pred_bagging)

print("Bagged Decision Metrics: ")
print(f'Accuracy: {bagging_accuracy}')
print(f'Precision: {bagging_precision}')
print(f'Recall: {bagging_recall}')
print(f'F1 Score: {bagging_f1}')
print('\n')
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
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)

print("Random Forest Metrics: ")
print(f'Accuracy: {rf_accuracy}')
print(f'Precision: {rf_precision}')
print(f'Recall: {rf_recall}')
print(f'F1 Score: {rf_f1}')

# Save the model
joblib.dump(random_forest_model, 'Exploratory Data Analysis\\random_forest_model.pkl')


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

# Train and evaluate Random Forest model (Cleaned)
random_forest_model_cleaned = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model_cleaned.fit(X_train_cleaned, y_train_cleaned)
y_pred_rf_cleaned = random_forest_model_cleaned.predict(X_test_cleaned)

rf_accuracy_cleaned = accuracy_score(y_test_cleaned, y_pred_rf_cleaned)
rf_precision_cleaned = precision_score(y_test_cleaned, y_pred_rf_cleaned)
rf_recall_cleaned = recall_score(y_test_cleaned, y_pred_rf_cleaned)
rf_f1_cleaned = f1_score(y_test_cleaned, y_pred_rf_cleaned)

print("Random Forest Metrics (Cleaned): ")
print(f'Accuracy: {rf_accuracy_cleaned}')
print(f'Precision: {rf_precision_cleaned}')
print(f'Recall: {rf_recall_cleaned}')
print(f'F1 Score: {rf_f1_cleaned}')
print('\n')

# Visualize outlier removal
def plot_boxplots(df, features, title):
    plt.figure(figsize=(10, 6))
    df[features].boxplot(rot=45, patch_artist=True, vert=False)
    plt.title(title)
    plt.show()

# Plotting boxplots before outlier removal
plot_boxplots(df, features, 'Box Plots Before Outlier Removal')

# Plotting boxplots after outlier removal
plot_boxplots(df_cleaned, features, 'Box Plots After Outlier Removal')

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


# CROSS-VALIDATION
# =================================================

from sklearn.model_selection import cross_val_score

# Cross-validation for Bagged Decision Tree
bagging_cv_scores = cross_val_score(bagging_model, X, y, cv=5, scoring='accuracy')
print(f'\nBagged Decision Tree Cross-Validation Accuracy: {bagging_cv_scores.mean()}')

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


# Feature Selection
# ============================================

# Select features with importance >0.06
selected_features = feature_importance_df[feature_importance_df['Importance'] > 0.06]['Feature'].tolist()

print(f'\nSelected Features: {selected_features}')

# Perform feature selection

X_selected = df[selected_features]

# Split data with selected features
X_train_selected, X_test_selected, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Train and evalute models with selected features
print('\n============== Evaluation after Feature Selection ============================')

# Train and evaluate Bagged Decision Tree with selected features
bagging_model_selected = BaggingClassifier(DecisionTreeClassifier(), n_estimators=50, random_state=42)
bagging_model_selected.fit(X_train_selected, y_train)
y_pred_bagging_selected = bagging_model_selected.predict(X_test_selected)
bagging_accuracy_selected = accuracy_score(y_test, y_pred_bagging_selected)
bagging_precision_selected = precision_score(y_test, y_pred_bagging_selected)
bagging_recall_selected = recall_score(y_test, y_pred_bagging_selected)
bagging_f1_selected = f1_score(y_test, y_pred_bagging_selected)

print("Bagged Decision Metrics: ")
print(f'Accuracy: {bagging_accuracy_selected}')
print(f'Precision: {bagging_precision_selected}')
print(f'Recall: {bagging_recall_selected}')
print(f'F1 Score: {bagging_f1_selected}')
print('\n')


# Train and evaluate Random Forest with selected features
random_forest_model_selected = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model_selected.fit(X_train_selected, y_train)
y_pred_rf_selected = random_forest_model_selected.predict(X_test_selected)
rf_accuracy_selected = accuracy_score(y_test, y_pred_rf_selected)
rf_precision_selected = precision_score(y_test, y_pred_rf_selected)
rf_recall_selected = recall_score(y_test, y_pred_rf_selected)
rf_f1_selected = f1_score(y_test, y_pred_rf_selected)

print("Random Forest Metrics: ")
print(f'Accuracy: {rf_accuracy_selected}')
print(f'Precision: {rf_precision_selected}')
print(f'Recall: {rf_recall_selected}')
print(f'F1 Score: {rf_f1_selected}')