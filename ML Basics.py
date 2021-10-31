# -*- coding: utf-8 -*-
"""
Created on Thu May 16 22:19:02 2019

@author: magic
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('voice.csv')
print(data.head())

data = data.iloc[:,:].values

#onehotencoder

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

LabelEncoder = LabelEncoder()
data[:,-1] = LabelEncoder.fit_transform(data[:,-1])

onehotencoder = OneHotEncoder(categorical_features=[-1])
data = onehotencoder.fit_transform(data).toarray()
print(data)

x = data[:,2:]
y = data[:,0]
print(x)
print(y)


#train test split
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size = 0.2, random_state = 0 )
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


#logistic model  
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

classifier.fit(x_train,y_train)      
y_pred = classifier.predict(x_test)

#accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_pred,y_test))

#feature engineering - choose feature through corr
data2 = pd.read_csv('voice.csv')
data2['label'] = data2['label'].map({'male':1,'female':0})
corr = data2.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr,cbar=True,square=True,cmap='coolwarm')
plt.show()

del data2['skew']
data2.drop('maxdom',axis=1,inplace=True)
data2.drop('centroid',axis=1,inplace=True)

#train ageain
x = data2.iloc[:,:-1]
y = data2['label']

x_train, x_test, y_train, y_test =train_test_split(x,y,test_size = 0.2, random_state = 0 )

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

classifier = LogisticRegression()

classifier.fit(x_train,y_train)      
y_pred = classifier.predict(x_test)

print(metrics.accuracy_score(y_pred,y_test))

