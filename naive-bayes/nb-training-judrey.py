# Import necessary libraries
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
import seaborn as sns

# Load the dataset
file_path = "C:/Users/jpads/Downloads/Thyroid Cancer Risk DatasetBang.csv"  # Replace with your file path
data = pd.read_csv(file_path)

# Preprocessing: Convert numerical columns to float
numerical_cols = ['Age', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']
# for col in numerical_cols:
#     data[col] = data[col].astype(float)

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

# Split the dataset into features (X) and target (y)
X = data.drop('Thyroid_Cancer_Risk', axis=1)
y = data['Thyroid_Cancer_Risk']

# Step 7: Split the dataset into 90-10 proportion
X_train_test, X_unseen, y_train_test, y_unseen = train_test_split(data, y, test_size=0.1, random_state=42)

# Step 8: Further split training & testing data into 80-20 proportion
X_train, X_test, y_train, y_test = train_test_split(X_train_test, y_train_test, test_size=0.2, random_state=42)

# Create and train the Gaussian Naive Bayes model
model = GaussianNB()

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
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=target_encoder.classes_)
roc_aoc= roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_encoder.classes_,
            yticklabels=target_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
print("ROC-AUC:")
print(roc_aoc)
print("Mean Cross-Validation Accuracy:", cv_scores.mean())

