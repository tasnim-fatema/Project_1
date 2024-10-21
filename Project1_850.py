import numpy as np
import pandas as pd
import joblib 
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.metrics import f1_score, precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report


df = pd.read_csv("C:/Users/16477/Documents/GitHub/aer850_f24/Project1Data.csv")

print(df.info())
print(df.columns)

# ----------------------------------------
# Step 1: Data Processing
# ----------------------------------------

# Define features (X, Y, Z) and target (Step)
# X is the DataFrame containing the 'X', 'Y', and 'Z' columns (features)
# y is the 'Step' column (target)
X = df[['X', 'Y', 'Z']]        
y = df['Step']

x_1 = df.get("X")
y_1 = df.get("Y")
z_1 = df.get("Z") 


# ----------------------------------------
#  Step 2: Data Visualization
# ----------------------------------------
# # Create a 3D scatter plot of the data using 'X', 'Y', and 'Z' values
# ax = plt.axes(projection='3d')
# ax.scatter3D(x_1,y_1,z_1, s = 10);
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

# Check the distribution of 'Step' to see if stratification is needed
print(y.value_counts())

# ----------------------------------------
#  Step 3: Correlation Analysis
# ----------------------------------------

# use StratifiedShuffleSplit() 
my_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Perform stratified splitting
for train_index, test_index in my_splitter.split(X, y):
    X_train = X.loc[train_index].reset_index(drop=True)
    X_test = X.loc[test_index].reset_index(drop=True)
    y_train = y.loc[train_index].reset_index(drop=True)
    y_test = y.loc[test_index].reset_index(drop=True)

# Display the shape of the resulting datasets
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Training target shape: {y_train.shape}")
print(f"Testing target shape: {y_test.shape}")

#Visualize the correlation matrix
df.corr()
sns.heatmap(df.corr().round(2), annot=True, cmap="magma")
corr1 = y_train.corr(X_train['X'])
print(corr1)
corr2 = y_train.corr(X_train['Y'])
print(corr2)
corr3 = y_train.corr(X_train['Z'])
print(corr3)
plt.show

# 
# ----------------------------------------
# Step 4 & 5: Classification Model Development + Model Performance Analysis
# ----------------------------------------


# ----------------------------------------
# SVM with RandomizedSearchCV
# ----------------------------------------
svm_model = SVC()  #Support Vector Classifier for classification tasks
param_dist_svm = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf']
}
random_search_svm = RandomizedSearchCV(svm_model, param_distributions=param_dist_svm, n_iter=10, cv=5, random_state=42, n_jobs=-1)
random_search_svm.fit(X_train, y_train)
best_model_svm = random_search_svm.best_estimator_  # This was previously referred to as best_model_svm
print("Best SVM Model:", best_model_svm)

# Predictions and evaluation
y_pred_svm = best_model_svm.predict(X_test)
print(f"Accuracy (SVM): {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"F1 Score (SVM): {f1_score(y_test, y_pred_svm, average='weighted'):.5f}")
print(classification_report(y_test, y_pred_svm))

# train_accuracy_svm = accuracy_score(y_train, y_train_pred_svm)
# test_accuracy_svm = accuracy_score(y_test, y_pred_svm)
# print(f"SVM Training Accuracy: {train_accuracy_svm:.4f}")
# print(f"SVM Testing Accuracy: {test_accuracy_svm:.4f}")

# Training predictions and accuracy for SVM
y_train_pred_svm = best_model_svm.predict(X_train)
train_accuracy_svm = accuracy_score(y_train, y_train_pred_svm)
print(f"SVM Training Accuracy: {train_accuracy_svm:.4f}")

# Test predictions and accuracy for SVM
test_accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Testing Accuracy: {test_accuracy_svm:.4f}")

# ----------------------------------------
# Decision Tree Classification
# ----------------------------------------
decision_tree = DecisionTreeClassifier(random_state=24)
param_grid_dt = {
    'max_depth': [None, 10, 20],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4]
}
grid_search_dt = GridSearchCV(decision_tree, param_grid_dt, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_dt.fit(X_train, y_train)
best_model_dt = grid_search_dt.best_estimator_
print("Best Decision Tree Model:", best_model_dt)

# Make predictions and evaluate Decision Tree model
y_pred_dt = best_model_dt.predict(X_test)
print(f"Accuracy (Decision Tree): {accuracy_score(y_test, y_pred_dt):.2f}")
f1_dt = f1_score(y_test, y_pred_dt, average='weighted')
print(f"F1 Score (Decision Tree): {f1_dt:.2f}")
# Assuming best_model_dt is your trained Decision Tree model
y_train_pred_dt = best_model_dt.predict(X_train)
train_accuracy_dt = accuracy_score(y_train, y_train_pred_dt)
print(f"Decision Tree Training Accuracy: {train_accuracy_dt:.4f}")


# Generate and plot the confusion matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_dt)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Decision Tree")
plt.show()

# Test predictions and accuracy for Decision Tree
y_pred_dt = best_model_dt.predict(X_test)
test_accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Testing Accuracy: {test_accuracy_dt:.4f}")

# ----------------------------------------
# Random Forest Classification
# ----------------------------------------
random_forest = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [10, 30, 50],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
grid_search_rf = GridSearchCV(random_forest, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
best_model_rf = grid_search_rf.best_estimator_
print("Best Random Forest Model:", best_model_rf)

# Make predictions and evaluate Random Forest model
y_pred_rf = best_model_rf.predict(X_test)
print(f"Accuracy (Random Forest): {accuracy_score(y_test, y_pred_rf):.2f}")
f1 = f1_score(y_test, y_pred_rf, average='weighted')
print(f"F1 Score (Random Forest): {f1:.2f}")
print(classification_report(y_test, y_pred_rf))

# Assuming best_model_rf is your trained Random Forest model
y_train_pred_rf = best_model_rf.predict(X_train)
train_accuracy_rf = accuracy_score(y_train, y_train_pred_rf)
print(f"Random Forest Training Accuracy: {train_accuracy_rf:.4f}")

# Test predictions and accuracy for Random Forest
y_pred_rf = best_model_rf.predict(X_test)
test_accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Testing Accuracy: {test_accuracy_rf:.4f}")




# Print classification report for more details
print("Classification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))

# # Generate the confusion matrix
# cm = confusion_matrix(y_test, y_pred_rf)

# # Define the class labels (e.g., 'Step' column values)
# class_labels = sorted(y.unique())  # Get the sorted unique classes

# # Plot the confusion matrix
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
# disp.plot(cmap=plt.cm.Blues)

# # Add title and customizations
# plt.title('Confusion Matrix (Random Forest)', fontsize=15, pad=20)
# plt.xlabel('Predicted Labels', fontsize=9)
# plt.ylabel('True Labels', fontsize=10)

# ----------------------------------------
# Logistic Regression
# ----------------------------------------

model = LogisticRegression(max_iter=1000)  # Ensure 'saga' solver for elasticnet
param_grid_lr = [
    {
          # 'none' is removed as it's not supported by all solvers
        'C': [0.01, 0.1, 1, 10, 100],
        'max_iter': [2000]
    }
]

clf = GridSearchCV(model, param_grid=param_grid_lr, cv=3, n_jobs=-1)
best_clf = clf.fit(X_train, y_train)  # Make sure to use the correct variables

# Use the best estimator to make predictions
y_pred = best_clf.best_estimator_.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Best Logistic Regression Model:", best_clf)
best_model_lr= clf.best_estimator_
print("Best Logistic Regression Model:", best_model_lr)

# Evaluate the model
print("\nClassification Report (Logistic Regression):\n", classification_report(y_test, y_pred))

# Assuming best_model_lr is your trained Logistic Regression model
y_train_pred_lr = best_model_lr.predict(X_train)
train_accuracy_lr = accuracy_score(y_train, y_train_pred_lr)
print(f"Logistic Regression Training Accuracy: {train_accuracy_lr:.4f}")

# Test predictions and accuracy for Logistic Regression
y_pred_lr = best_model_lr.predict(X_test)
test_accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Testing Accuracy: {test_accuracy_lr:.4f}")

# ----------------------------------------
# Step 6: Stacked Model Performance Analysis
# ----------------------------------------

#StackingClassifier
base_learners = [
    ('random_forest', best_model_rf),  # Replace with your best RandomForest model
    ('svm', best_model_svm)            # Replace with your best SVM model
]

# Logistic Regression as the final estimator
final_estimator = LogisticRegression(max_iter=1000)

# Create the stacking classifier
stacked_model = StackingClassifier(estimators=base_learners, final_estimator=final_estimator, cv=5)

# Train the stacking model
stacked_model.fit(X_train, y_train)

# Predictions on the test set
y_pred_stacked = stacked_model.predict(X_test)

# Evaluate the model
f1 = f1_score(y_test, y_pred_stacked, average='weighted')
precision = precision_score(y_test, y_pred_stacked, average='weighted')
accuracy = accuracy_score(y_test, y_pred_stacked)

# Print performance metrics
print(f"F1 Score (Stacked Model): {f1:.2f}")
print(f"Precision (Stacked Model): {precision:.2f}")
print(f"Accuracy (Stacked Model): {accuracy:.2f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_stacked)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Stacked Model')
plt.show()


# ----------------------------------------
# Step 7: Model Evaluation
# ----------------------------------------
# Save the best model using joblib
joblib.dump(stacked_model, 'stacked_model.joblib')

# Load the saved model
stacked_model = joblib.load('stacked_model.joblib')

# Given coordinates to predict the maintenance step
new_coordinates = np.array([
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3, 1.8],
    [9.4, 3, 1.3]
])

# # Predict the maintenance step using the loaded model
# predicted_steps = stacked_model.predict(new_coordinates)

# # Print the predicted maintenance steps
# print("Predicted Maintenance Steps for the provided coordinates:", predicted_steps)


# Convert the new coordinates to a DataFrame with feature names
new_coordinates_df = pd.DataFrame(new_coordinates, columns=['X', 'Y', 'Z'])

# Predict the maintenance step using the loaded model
predicted_steps = stacked_model.predict(new_coordinates_df)

# Print the predicted maintenance steps
print("Predicted Maintenance Steps for the provided coordinates:", predicted_steps)   
          
        




