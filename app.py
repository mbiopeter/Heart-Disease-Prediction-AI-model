"""importing the depedencies"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""Data collection and processing"""
#loading the csv file into pandas Dataframe
heart_data = pd.read_csv('D:/Projects/machine learning/heart deseases prediction/data.csv')
#get the first five rows of the dataset
heart_data.head()
#get the last five rows of the dataset
heart_data.tail()
#check the number of rows and columns in the dataset
heart_data.shape
#check the data types of each column
heart_data.info()
#check for data redandancy
heart_data.isnull().sum()
#get the statistical measure of the data
heart_data.describe()
#check the distribution of target variable
heart_data['target'].value_counts()
    #1 --> Defective heart
    #2 --> Healthy heart

"""splitting features and the target"""
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

"""Splitting data into training data and test data"""
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

"""Training the model"""
"""Logic Regression"""
model = LogisticRegression()
#train logistic regression model with training data
model.fit(X_train, Y_train)

"""Model evaluation"""
"""Accuracy score"""
#Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
#print("Accuracy on training data : ", training_data_accuracy )
#Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
#print("Accuracy on test data : ", test_data_accuracy )

"""Buld a predictive system"""
input_data = (67,1,2,152,212,0,0,150,0,0.8,1,0,3)
#change the input data into a numpy array
input_data_as_numpy_array  = np.asarray(input_data)
#reshape the numpy array as we are predicting only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)

if prediction[0] == 0:
    print("The person is healthy")
else:
    print("The person is heart defective")