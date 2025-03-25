# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df=pd.read_csv('creditcard.csv')


# Impute missing values with the mean of the column
df['V23'].fillna(df['V23'].mean(), inplace=True)
df['V24'].fillna(df['V24'].mean(), inplace=True)
df['V25'].fillna(df['V25'].mean(), inplace=True)
df['V26'].fillna(df['V26'].mean(), inplace=True)
df['V27'].fillna(df['V27'].mean(), inplace=True)
df['V28'].fillna(df['V28'].mean(), inplace=True)
df['Amount'].fillna(df['Amount'].mean(), inplace=True)
df['Class'].fillna(df['Class'].mean(), inplace=True)

df['Class'] = df['Class'].astype(int)



#Data is imbalanced as 0---> Normal data is Huge and 1--->Fraud Data is Low
#So it will predict every transaction as normal one .

# Hence , Splitting data into 0 and 1 for analysis

LegitData = df[df.Class==0]
FraudData = df[df.Class==1]


#Build a samle dataset of Legit Transaction and Fraud Transaction
#Fraud Transaction are 103
#We will choose 103 random Transaction from legit transaction and join in legit data
#This makes a uniformly distributed dataset which is good for modelling

Legit_sample = LegitData.sample(n=103)

#Concantetanate 2 DataFrame

newdf = pd.concat([Legit_sample,FraudData],axis=0)


#Compare Values for both Transaction
#If values are highly different then it is a bad sample
#If values are almost same then it is a good sample
#Below we are getting almost similiar data as of the old imbalanced dataset and newly created dataset


#Split Data into Features and Target
# Axis=0 represent ROW and Axis=1 represent COLUMN

X = newdf.drop(columns='Class' , axis=1)
y = newdf['Class']


#Split Data into Training Data and Testing Data

X_train , X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=2)

#Model Training

model = LogisticRegression(max_iter=1000)

#Training the model with training data

model.fit(X_train , y_train)

#Model Evaluation for Training Data

#Accuracy Score
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction , y_train)

print("Accuracy on Training Data : " , training_data_accuracy )

#Model Evaluation for Testing Data

#Accuracy Score
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction , y_test)

print("Accuracy on Testing Data : " , testing_data_accuracy )