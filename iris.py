import numpy as np  #linear algebra
import pandas as pd   #data processing
df =pd.read_csv("Iris.csv")
df

df.shape

#column dropping
df=df.drop(columns=["Id"])
df
df=df.drop(columns=["Species"])
df

df.head() #Return first 5 entries

df["Species"].replace({"Iris-setosa":1,"Iris-versicolor":2,"Iris-virginica":3},inplace=True)
df

y=df["Species"].replace({"Iris-setosa":1,"Iris-versicolor":2,"Iris-virginica":3},inplace=True)
y

c=df.Species.values.reshape(-1,1)
c

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)
x_train.shape
y_train.shape


p=6
knclr=KNeighborsClassifier(p)

knclr

knclr.fit(x_train,y_train)

y_pred=knclr.predict(x_test)

y_pred=knclr.predict(x_test)
