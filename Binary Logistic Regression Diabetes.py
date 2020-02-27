#Let's import the necessary Libraries
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error 


#Let's Read the data into pandas Dataframe
diabetes = pd.read_csv('diabetes.csv')

#Let's Analyze the Data
print(diabetes.head())
print(diabetes.columns)
print(diabetes.describe)

#We Select the features: independent variable(x) & dependent variable(y)
#We wish to predict the outcome of a person having Diabetes or Not
#Hence we select Outcome as our dependent variable(y)

x = diabetes.iloc[: , :8]
y = diabetes['Outcome']

#Now, we will split our data into 2 categories:Training & validation data
train_x,val_x,train_y,val_y = train_test_split(x,y,train_size=0.7,random_state=1)

#Let's build our Machine Learning Model
diabetesModel = LogisticRegression()

#We train the model using our Training Data
diabetesModel.fit(train_x,train_y)

#It's Time to predict the total_marks using our model
prediction = diabetesModel.predict(val_x)

#Let's evaluate the error
error = mean_squared_error(val_y,prediction)
print(error)

#TO BE CONTINUED...!!!