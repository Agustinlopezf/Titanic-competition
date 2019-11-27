import csv
import os
import re
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.stats.stats import pearsonr
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


data = pd.read_csv('train.csv')

#Change the sex values by one and minus one
data.Sex[data.Sex == "male"] = 1.
data.Sex[data.Sex == "female"] = -1.


#Set the train and target variables
y = data.Survived
model_parameters= ['Sex', 'Age', 'Pclass', 'Fare']
X = data[model_parameters]



#Use LabelEncoder instead of one-hot encoder for embarked data
data.Embarked.fillna('Not available', inplace = True)
embarked_encoder = LabelEncoder()
embarked_encoder.fit(data.Embarked)
print(embarked_encoder.classes_)
X['Embarked'] = embarked_encoder.transform(data.Embarked)

#Create a column to compute for the total family members
X['family-members'] = data['Parch'] + data['SibSp'] + 1

#Mean Age and Fare, which will be used to replace NaN values (both for train and test data)
mean_age = np.mean(X.Age)
mean_fare = np.mean(X.Fare)

#Replace missing values with the mean
X.Age.fillna(mean_age, inplace = True)
X.Fare.fillna(mean_fare, inplace = True)


#Obtain the title from the name using regular expressions, and convert it to one-hot encoding
pattern = r'([A-Za-z])+, (\w+).'
data['Title'] = data['Name'].apply(lambda x: re.search(pattern, x).group(2))

data['Title'] = data['Title'].replace(['Mlle', 'Ms'], 'Miss')
data['Title'] = data['Title'].replace('Mme', 'Mrs')
data['Title'] = data['Title'].replace('the', 'Rare')
data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Master'],'Rare')

title_encoder = LabelEncoder()
title_encoder.fit(data['Title'])
print(title_encoder.classes_)
X['Title'] = title_encoder.transform(data.Title)


#Create age bins
age_bins = [0,18, 25, 35, 45, 100]
age_bins_labels = ['0-18', '18-25','25-35', '35-45', '45-100']

X['Age-bins'] = pd.cut(X.Age, bins = age_bins, labels = age_bins_labels)
print(X['Age-bins'].value_counts())

#Encode the age-bins
age_bins_encoder = LabelEncoder()
age_bins_encoder.fit(X['Age-bins'])
X['Age-bins'] = age_bins_encoder.transform(X['Age-bins'])



#Create fare bins using qcut so to obtain equally sized bins. Store the bins in fare_bins for use in test set
fare_bins_labels = ['1st quartile', '2nd quartile', '3rd quartile', '4rd quartile']

X['Fare-bins'], fare_bins = pd.qcut(X.Fare, q = 4, labels = fare_bins_labels, retbins = True)

print(X['Fare-bins'].value_counts())

fare_bins_encoder = LabelEncoder()
fare_bins_encoder.fit(X['Fare-bins'])
X['Fare-bins'] = fare_bins_encoder.transform(X['Fare-bins'])

X.drop(['Age', 'Fare'], axis = 1, inplace = True) #Drop the Age and Fare columns and keep the bins


#Split train and test data and train the estimator
train_X, test_X, train_y, test_y = train_test_split(X, y,random_state = 0)

data_model = GradientBoostingClassifier(n_estimators = 300, learning_rate = 0.15, min_samples_split = 5, min_samples_leaf = 5)
data_model.fit(train_X, train_y)
predicted_survival = data_model.predict(test_X)
print('The accuracy of the model is ' + str(accuracy_score(data_model.predict(test_X).round(), test_y)))

#Load and adjust submission data
data_to_predict = pd.read_csv('test.csv')


data_to_predict.Sex[data_to_predict.Sex == "male"] = 1.
data_to_predict.Sex[data_to_predict.Sex == "female"] = -1.

X_to_predict = data_to_predict[model_parameters]
X_to_predict['family-members'] = data_to_predict['Parch'] + data_to_predict['SibSp'] + 1
X_to_predict['Embarked'] = embarked_encoder.transform(data_to_predict.Embarked)

#Replace notna values of the Age parameter with the mean value. 
X_to_predict.Age.fillna(mean_age, inplace = True)
X_to_predict.Fare.fillna(mean_fare, inplace = True)


#Obtain title labels
data_to_predict['Title'] = data_to_predict['Name'].apply(lambda x: re.search(pattern, x).group(2))

data_to_predict['Title'] = data_to_predict['Title'].replace(['Mlle', 'Ms'], 'Miss')
data_to_predict['Title'] = data_to_predict['Title'].replace('Mme', 'Mrs')
data_to_predict['Title'] = data_to_predict['Title'].replace('the', 'Rare')
data_to_predict['Title'] = data_to_predict['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Master'],'Rare')

X_to_predict['Title'] = title_encoder.transform(data_to_predict.Title)

#Age and fare bins
X_to_predict['Age-bins'] = pd.cut(X_to_predict.Age, bins = age_bins, labels = age_bins_labels)
X_to_predict['Age-bins'] = age_bins_encoder.transform(X_to_predict['Age-bins'])



#Use the quartile bins obtained from the training data
X_to_predict['Fare-bins'] = pd.cut(X_to_predict.Fare, bins = fare_bins, labels = fare_bins_labels)
X_to_predict['Fare-bins'].fillna(fare_bins_labels[0], inplace = True) #Put the nan values in the first quartile
X_to_predict['Fare-bins'] = fare_bins_encoder.transform(X_to_predict['Fare-bins'])


X_to_predict.drop(['Age', 'Fare'], axis = 1, inplace = True) #Drop the Age and Fare columns and keep the bins

results = data_model.predict(X_to_predict)

#Convert to DataFrame and add the PassengerId 
output_dict = {'PassengerId': data_to_predict.PassengerId, 'Survived': results.astype(int)} #Convert the survived column to integers
output_dataframe = pd.DataFrame(output_dict)

#Save results as csv file
output_dataframe.to_csv('predictions_submission.csv', index = False)

