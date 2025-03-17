import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, make_scorer

# Load the dataset
file_path = "Downloads/Thyroid.csv"
data = pd.read_csv(file_path)

# Preprocessing: Convert numerical columns to float
numerical_cols = ['Age', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']
data[numerical_cols] = data[numerical_cols].astype(float)

# Encode categorical variables using LabelEncoder
categorical_cols = [col for col in data.columns if col not in numerical_cols and col != 'Thyroid_Cancer_Risk']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Save encoder for future use

# Encode the target column (Thyroid_Cancer_Risk)
target_encoder = LabelEncoder()
data['Thyroid_Cancer_Risk'] = target_encoder.fit_transform(data['Thyroid_Cancer_Risk'])

# Define features and target variable
X = data.drop(columns=["Thyroid_Cancer_Risk"], errors='ignore')
y = data["Thyroid_Cancer_Risk"]

# Split into 90% train-test and 10% unseen
X_tt, X_un, y_tt, y_un = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)

# Feature Selection using Mutual Information
selector = SelectKBest(score_func=mutual_info_classif, k=min(10, X_tt.shape[1]))
X_tt_sel = selector.fit_transform(X_tt, y_tt)
X_un_sel = selector.transform(X_un)

# Split train-test into 80% training and 20% testing
X_tr, X_te, y_tr, y_te = train_test_split(X_tt_sel, y_tt, test_size=0.20, random_state=42, stratify=y_tt)

# Apply feature scaling
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_te = scaler.transform(X_te)
X_un = scaler.transform(X_un_sel)

# Define the SVM model
svm = SVC(probability=True, kernel='rbf', random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.1, 1]}
grid_search = GridSearchCV(svm, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_tr, y_tr)

# Best model from tuning
best_svm = grid_search.best_estimator_

# Train the final model on the full training set
best_svm.fit(X_tr, y_tr)

# Evaluate on the test set
y_te_pred = best_svm.predict(X_te)
y_te_prob = best_svm.predict_proba(X_te)
acc_te = accuracy_score(y_te, y_te_pred)
prec_te = precision_score(y_te, y_te_pred, average="weighted")
rec_te = recall_score(y_te, y_te_pred, average="weighted")
roc_te = roc_auc_score(y_te, y_te_prob, multi_class="ovr")
conf_te = confusion_matrix(y_te, y_te_pred)

# Evaluate on the unseen set
y_un_pred = best_svm.predict(X_un)
y_un_prob = best_svm.predict_proba(X_un)
acc_un = accuracy_score(y_un, y_un_pred)
prec_un = precision_score(y_un, y_un_pred, average="weighted")
rec_un = recall_score(y_un, y_un_pred, average="weighted")
roc_un = roc_auc_score(y_un, y_un_prob, multi_class="ovr")
conf_un = confusion_matrix(y_un, y_un_pred)

# Define labels
labels = ["Low", "Medium", "High"]

# Print Final Model Evaluation Results
test_results = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "ROC-AUC"],
    "Testing Data": [acc_te, prec_te, rec_te, roc_te]
})

unseen_results = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "ROC-AUC"],
    "Unseen Data": [acc_un, prec_un, rec_un, roc_un]
})

# Plot confusion matrix for test data
plt.figure(figsize=(6,5))
sns.heatmap(conf_te, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM Confusion Matrix for Testing Data')
plt.show()

print("\nModel Evaluation Test Results:")
print(test_results.to_string(index=False))

# Plot confusion matrix for unseen data
plt.figure(figsize=(6,5))
sns.heatmap(conf_un, annot=True, fmt='d', cmap='Oranges', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM Confusion Matrix for Unseen Data')
plt.show()

print("\nModel Evaluation Unseen Results:")
print(unseen_results.to_string(index=False))
