import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score

# Step 1: Load the dataset
data = pd.read_csv("C:/Users/eugen/OneDrive/Desktop/Thyroid Cancer Risk Dataset.csv")

# Step 2: Split into 90% (training & testing) and 10% (unseen data)
train_test_data = data.sample(frac=0.9, random_state=42)
unseen_data = data.drop(train_test_data.index)

# Step 3: Split the 90% data into 80% training and 20% testing
X = train_test_data.iloc[:, :-1]  # Features
y = train_test_data.iloc[:, -1]   # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Preprocess the data (handle categorical features)
categorical_cols = X_train.select_dtypes(include=['object']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols)  # One-hot encode categorical columns
    ],
    remainder='passthrough'  # Leave numerical columns unchanged
)

# Step 5: Build the Naive Bayes model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GaussianNB())
])

# Step 6: Perform 10-fold cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation accuracy:", cv_scores.mean())

# Get predictions for confusion matrix and other metrics
y_pred = cross_val_predict(model, X_train, y_train, cv=10)

# Step 7: Train the model on the full training set
model.fit(X_train, y_train)

# Step 8: Evaluate on the testing set
y_test_pred = model.predict(X_test)
y_test_pred_proba = model.predict_proba(X_test)

# Confusion Matrix
test_conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Test Confusion Matrix:\n", test_conf_matrix)

# Plot confusion matrix for testing set
plt.figure(figsize=(8, 6))
sns.heatmap(test_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.title('Confusion Matrix (Testing Set)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)

# Precision (macro)
test_precision = precision_score(y_test, y_test_pred, average='macro')
print("Test Precision (macro):", test_precision)

# Recall (macro)
test_recall = recall_score(y_test, y_test_pred, average='macro')
print("Test Recall (macro):", test_recall)

# ROC-AUC (macro)
test_roc_auc = roc_auc_score(y_test, y_test_pred_proba, multi_class='ovr', average='macro')
print("Test ROC-AUC (macro):", test_roc_auc)

# Step 9: Validate on unseen data
X_unseen = unseen_data.iloc[:, :-1]
y_unseen = unseen_data.iloc[:, -1]

# Evaluate on unseen data
y_unseen_pred = model.predict(X_unseen)
y_unseen_pred_proba = model.predict_proba(X_unseen)

# Confusion Matrix
unseen_conf_matrix = confusion_matrix(y_unseen, y_unseen_pred)
print("Unseen Confusion Matrix:\n", unseen_conf_matrix)

# Plot confusion matrix for unseen data
plt.figure(figsize=(8, 6))
sns.heatmap(unseen_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.title('Confusion Matrix (Unseen Data)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Accuracy
unseen_accuracy = accuracy_score(y_unseen, y_unseen_pred)
print("Unseen Accuracy:", unseen_accuracy)

# Precision (macro)
unseen_precision = precision_score(y_unseen, y_unseen_pred, average='macro')
print("Unseen Precision (macro):", unseen_precision)

# Recall (macro)
unseen_recall = recall_score(y_unseen, y_unseen_pred, average='macro')
print("Unseen Recall (macro):", unseen_recall)

# ROC-AUC (macro)
unseen_roc_auc = roc_auc_score(y_unseen, y_unseen_pred_proba, multi_class='ovr', average='macro')
print("Unseen ROC-AUC (macro):", unseen_roc_auc)