import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB, CategoricalNB, MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, recall_score, \
    precision_score
import seaborn as sns

# Load the dataset
file_path = "Thyroid Cancer Risk Dataset Clean.csv"
data = pd.read_csv(file_path)

# Define categorical and numerical features (keep this part unchanged)
categorical_features = ['Gender', 'Family_History', 'Radiation_Exposure', 'Iodine_Deficiency', 'Smoking', 'Obesity', 'Diabetes']
numerical_features = ['Age', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']

# Create a pipeline for numerical features
numerical_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('discretizer', KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile'))
])

# Update the ColumnTransformer to include the numerical pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),  # Apply standardization then discretization
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Keep the rest of your main pipeline unchanged
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', CategoricalNB()),
])

# Encode the target column (Thyroid_Cancer_Risk)
target_encoder = LabelEncoder()
data['Thyroid_Cancer_Risk'] = target_encoder.fit_transform(data['Thyroid_Cancer_Risk'])

# Split the dataset into features (X) and target (y)
X = data.drop('Thyroid_Cancer_Risk', axis=1)
y = data['Thyroid_Cancer_Risk']

# Split the dataset into 90-10 proportion
X_train_test, X_unseen, y_train_test, y_unseen = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# Further split training & testing data into 80-20 proportion
X_train, X_test, y_train, y_test = train_test_split(X_train_test, y_train_test, test_size=0.2, random_state=42, stratify=y_train_test)

# Apply 10-fold cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Get probability estimates for each class
y_pred_proba = model.predict_proba(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
test_conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=target_encoder.classes_)
roc_auc= roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(test_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_encoder.classes_,
            yticklabels=target_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Using Test Data)')
plt.tight_layout()
plt.savefig('test_confusion_matrix.png')
plt.show()

# Print the results for evaluation on test data
print("Mean Cross-Validation Accuracy (Using Test Data):", cv_scores.mean())
print("Confusion Matrix (Using Test Data):")
print(test_conf_matrix)
# Accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_accuracy)
# Precision (macro)
test_precision = precision_score(y_test, y_pred, average='macro')
print("Test Precision (macro):", test_precision)
# Recall (macro)
test_recall = recall_score(y_test, y_pred, average='macro')
print(f"Test Recall (macro): {test_recall}")
# ROC-AUC (macro)
print(f"Test ROC-AUC (macro): {roc_auc}")

# Predict on unseen data
unseen_pred = model.predict(X_unseen)
unseen_accuracy = accuracy_score(y_unseen, unseen_pred)
unseen_pred_proba = model.predict_proba(X_unseen)
unseen_conf_matrix = confusion_matrix(y_unseen, unseen_pred)
unseen_recall = recall_score(y_unseen, unseen_pred, average='macro')
unseen_precision = precision_score(y_unseen, unseen_pred, average='macro')
unseen_class_report = classification_report(y_unseen, unseen_pred, target_names=target_encoder.classes_)
unseen_roc_auc= roc_auc_score(y_unseen, unseen_pred_proba, multi_class='ovr')


# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(unseen_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_encoder.classes_,
            yticklabels=target_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Using Unseen Data)')
plt.tight_layout()
plt.savefig('unseen_confusion_matrix.png')
plt.show()

print("Confusion Matrix (Using Unseen Data):")
print(unseen_conf_matrix)
# Accuracy
print("Unseen Data Accuracy:", unseen_accuracy)
# Precision (macro)
print("Unseen Data Precision (macro):", test_precision)
# Recall (macro)
print(f"Unseen Data Recall (macro): {test_recall}")
# ROC-AUC (macro)
print(f"Unseen Data ROC-AUC (macro): {unseen_roc_auc}")
