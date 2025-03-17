# Import needed libraries
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from mord import LogisticAT
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix

# Step 1: Load the dataset
df = pd.read_csv('/Users/acengaosi/Downloads/Thyroid Cancer Risk Dataset - RESTRICT.csv')

# Step 2: Split into 90% (training and testing) and 10% (unseen data)
training_data = df.sample(frac=0.9, random_state=42)
unseen_data = df.drop(training_data.index)

#Step 3: Split training and testing data
X = training_data.drop('Thyroid_Cancer_Risk', axis=1)
y = training_data['Thyroid_Cancer_Risk']

y = y.map({'Low': 0, 'Medium': 1, 'High': 2}) # Put labels
class_labels = ['Low', 'Medium', 'High']

# Step 4: Preprocess data (handle categorical and numerical data)
categorical_data = X.select_dtypes(include=['object']).columns
numerical_data = X.select_dtypes(include=['float', 'int']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_data),
        ('cat', OneHotEncoder(), categorical_data)
    ])

# Step 5: Build Ordinal Logistic Regression Model
olrm = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticAT())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Perform 10-fold cross validation
cv_scores = cross_val_score(olrm, X_train, y_train, cv=10, scoring='accuracy')
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation accuracy:", cv_scores.mean())

# Step 7: Train the model
olrm.fit(X_train, y_train)

# Step 8: Evaluate on the testing set
y_pred = olrm.predict(X_test)

# Calculate accuracy, precision (macro), and recall (macro)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
print("Accuracy:", accuracy)
print("Precision (Macro):", precision)
print("Recall (Macro):", recall)

# Calculate ROC-AUC (One-vs-Rest approach)
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
y_pred_proba = olrm.predict_proba(X_test)

roc_auc = roc_auc_score(y_test_binarized, y_pred_proba, multi_class='ovr', average='macro')
print("ROC-AUC (OvR):", roc_auc)

# Plot confusion matrix for training and testing set
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',  xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('OLRM Confusion Matrix (Training and Testing)')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Step 9: Validate on unseen data
X_unseen = unseen_data.drop('Thyroid_Cancer_Risk', axis=1)
y_unseen = unseen_data['Thyroid_Cancer_Risk']
y_unseen = y_unseen.map({'Low': 0, 'Medium': 1, 'High': 2}) # Put labels

# Evaluate on unseen data
y_unseen_predicted = olrm.predict(X_unseen)

# Plot confusion matrix for unseen data
unseen_conf_matrix = confusion_matrix(y_unseen, y_unseen_predicted)
plt.figure(figsize=(8, 6))
sns.heatmap(unseen_conf_matrix, annot=True, fmt='d', cmap='Blues',  xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('OLRM Confusion Matrix (Unseen)')
plt.tight_layout()
plt.savefig('confusion_matrix_unseen.png')
plt.show()

# Calculate accuracy, precision (macro), and recall (macro)
accuracy_unseen = accuracy_score(y_unseen, y_unseen_predicted)
precision_unseen = precision_score(y_unseen, y_unseen_predicted, average='macro')
recall_unseen = recall_score(y_unseen, y_unseen_predicted, average='macro')
print("Accuracy:", accuracy_unseen)
print("Precision (Macro):", precision_unseen)
print("Recall (Macro):", recall_unseen)

# Calculate ROC-AUC (One-vs-Rest approach)
y_unseen_binarized = label_binarize(y_unseen, classes=[0, 1, 2])
y_unseen_predicted_proba = olrm.predict_proba(X_unseen)

roc_auc_unseen = roc_auc_score(y_unseen_binarized, y_unseen_predicted_proba, multi_class='ovr', average='macro')
print("ROC-AUC (OvR):", roc_auc_unseen)
