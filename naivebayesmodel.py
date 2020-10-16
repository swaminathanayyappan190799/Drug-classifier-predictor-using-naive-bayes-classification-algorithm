#Importing libraries
import numpy as np
import pandas as pd
import pickle

#Reading the data with the help of pandas.
data=pd.read_csv('F:/STUDY MATERIALS/Flask tutorials/drug200.csv')

#Dependent and Independent variable allocation
X=data.iloc[:,0:5].values
y=data.iloc[:,5].values

#Data Encoding/Data preprocessing using scikit learn
from sklearn.preprocessing import LabelEncoder
l1=LabelEncoder()
l2=LabelEncoder()
l3=LabelEncoder()
l4=LabelEncoder()
X[:,1]=l1.fit_transform(X[:,1])
X[:,2]=l2.fit_transform(X[:,2])
X[:,3]=l3.fit_transform(X[:,3])
y=l4.fit_transform(y)

#Performing onehotencoding for the bplev feature 
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X=X[:,1:] #Inorder to avoid dummy variable trap removing one of the feature of bpplev.

#Train and test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Naive bayes classifier implementation
from sklearn.naive_bayes import GaussianNB
nb_classifier=GaussianNB()
nb_classifier.fit(X_train,y_train)

#Converting the classifier model into a pickle file
pickle.dump(nb_classifier,open('model.pkl','wb'))

#Assigning the classifier model to a variable named as 'model'
model=pickle.load(open('model.pkl','rb'))