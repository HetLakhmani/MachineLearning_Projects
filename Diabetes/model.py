import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('diabetes.csv') 
df = pd.read_csv('diabetes.csv')

# printing the first 5 rows of the dataset
print(df.head())

# # number of rows and Columns in this dataset
print(df.shape)

# getting the statistical measures of the data
print(df.describe())

print(df['Outcome'].value_counts())

print(df.groupby('Outcome').mean())

# separating the data and labels
X = df.drop(columns = 'Outcome', axis=1)
Y = df['Outcome']

print(X)
print(Y)

# Data Standardization

scaler = StandardScaler()
scaler.fit(X)

standardized_data = scaler.transform(X)
print(standardized_data)

X = standardized_data
Y = df['Outcome']

print(X)
print(Y)

#Train Test Slipt
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Training the model
classifier = svm.SVC(kernel='linear')

#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)

# Accuracy Score

# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

# making a pickle file for the model

pickle.dump(classifier, open("model.pkl", "wb"))