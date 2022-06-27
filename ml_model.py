# importing Libraries
import pandas as pd
import numpy as np
import pickle

df=pd.read_csv('SUV_Purchase.csv')

# feature engineering
df=df.drop('User ID',axis=1)
df=df.drop('Gender',axis=1)

# loading the data
x1=np.array(df[['Age']])
x2=np.array(df[['EstimatedSalary']])
y=np.array(df[['Purchased']])
x=np.concatenate((x1,x2),axis=1)

#Splitting the data into Training Set and Testing Set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# standard scaling - normalizing the x_train
from sklearn.preprocessing import StandardScaler
sst=StandardScaler()
x_train=sst.fit_transform(x_train)

# training the model
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)

#Predicting the test set results
y_pred=model.predict(sst.transform(x_test))
print(y_pred)

#pickling
pickle.dump(model,open('model.pkl','wb')) # We are Serializing our model by creating model.pkl and writing it into it by wb
model=pickle.load(open('model.pkl','rb')) # deSerializing - reading the file rb
print("Success Loaded")

# Exexcute this file only once and create the pkl file


