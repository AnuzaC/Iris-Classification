#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv("iris.csv")

#Dependent variable is Y and Independent variable is X

X=dataset.iloc[:,1:-1].values
Y=dataset.iloc[:,-1].values



#Label Encoding and One Hot Encoding

from sklearn.preprocessing import LabelBinarizer
le= LabelBinarizer()
Y= le.fit_transform(Y)



#Dividing data into training set and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=1/3, random_state=0)


#Training dataset using Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train,Y_train)



#Prediction using Testing Set
Y_predict = classifier.predict(X_test)



#Checking Accuracy of Classification
from sklearn.metrics import confusion_matrix, accuracy_score
cm= confusion_matrix(Y_test.argmax(axis=1), Y_predict.argmax(axis=1))
acc = accuracy_score(Y_test.argmax(axis=1), Y_predict.argmax(axis=1))

