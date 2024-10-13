import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR


df = pd.read_csv("C:/Users/16477/Documents/GitHub/aer850_f24/Project1Data.csv")

print(df.info())
print(df.columns)

# Define features (X, Y, Z) and target (Step)
# X is the DataFrame containing the 'X', 'Y', and 'Z' columns (features)
# y is the 'Step' column (target)
X = df[['X', 'Y', 'Z']]        
y = df['Step']

# x_1 = df.get("X")
# y_1 = df.get("Y")
# z_1 = df.get("Z") 

# # Create a 3D scatter plot of the data using 'X', 'Y', and 'Z' values
# ax = plt.axes(projection='3d')
# ax.scatter3D(x_1,y_1,z_1, s = 10);
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

# Check the distribution of 'Step' to see if stratification is needed
print(y.value_counts())



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


# # Split the data into 80% training and 20% testing
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Display the shape of the resulting datasets
# print(f"Training data shape: {X_train.shape}")
# print(f"Testing data shape: {X_test.shape}")
# print(f"Training target shape: {y_train.shape}")
# print(f"Testing target shape: {y_test.shape}")

# Initialize the StandardScaler
my_scaler = StandardScaler()

# Fit the scaler on the training data and transform both the training and test data
X_train_scaled = my_scaler.fit_transform(X_train)
X_test_scaled = my_scaler.transform(X_test)


# Display the shape of the resulting datasets
print(f"Scaled Training data shape: {X_train_scaled.shape}")
print(f"Scaled Testing data shape: {X_test_scaled.shape}")

df.corr()
sns.heatmap(df.corr().round(2), annot=True)
corr1 = y_train.corr(X_train['X'])
print(corr1)
corr2 = y_train.corr(X_train['Y'])
print(corr2)
corr3 = y_train.corr(X_train['Z'])
print(corr3)

#Linear Regression
linear_reg = LinearRegression()
param_grid_lr = {} 
grid_search_lr = GridSearchCV(linear_reg, param_grid_lr, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_lr.fit(X_train, y_train)
best_model_lr = grid_search_lr.best_estimator_
print("Best Linear Regression Model:", best_model_lr)



#Support Vector Machine (SVM)
svr = SVR()




    
          
        




