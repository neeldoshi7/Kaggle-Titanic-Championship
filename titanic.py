
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('dataframe.csv')
X = dataset.iloc[:, 0:].values
y = dataset.iloc[:, 1].values
df = pd.DataFrame(X)

#for handling missing data

df.describe()
df.info()

mean_value = dataset['Age'].mean()
dataset['Age']=dataset['Age'].fillna(mean_value)

#Handling categorical data
# encode categorical data
dummyNew = pd.get_dummies(dataset['Embarked'])
dummyNew.head()

df = pd.concat([df , dummyNew], axis = 1)

df.to_csv (r'C:\Users\nldos\Desktop\Titanic\dataframe.csv', index = False, header=True)

#Splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)


# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

y_pred = regressor.predict(Xnew)

dff = pd.DataFrame(y_pred)

dff.to_csv (r'C:\Users\nldos\Desktop\Titanic\ansRF.csv', index = False, header=True)


#Model - Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

clf = DecisionTreeClassifier()

clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


#Graph

#Final Test 
datasetNew = pd.read_csv('test.csv')

dfNew = pd.DataFrame(Xnew)

dummyNew2 = pd.get_dummies(datasetNew['Embarked'])
dummyNew2.head()

dfNew = pd.concat([dfNew , dummyNew2], axis = 1)

mean_value = datasetNew['Age'].mean()
datasetNew['Age']=datasetNew['Age'].fillna(mean_value)

mean_value1 = datasetNew['Fare'].mean()
datasetNew['Fare']=datasetNew['Fare'].fillna(mean_value1)

dfNew.to_csv (r'C:\Users\nldos\Desktop\Titanic\dataframe2.csv', index = False, header=True)

datasett = pd.read_csv('dataframe2.csv')
Xnew = datasett.iloc[:, 1:].values

y_pred_new = clf.predict(Xnew)
dff = pd.DataFrame(y_pred_new)
dff.to_csv (r'C:\Users\nldos\Desktop\Titanic\ans.csv', index = False, header=True)
