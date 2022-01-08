import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,confusion_matrix

df = pd.read_csv('Clean_Train.csv')

Y = df['Default']
# print(df['Default'])
X = df.drop(columns= ['Default'])

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.25)
# print(x_train.shape,x_test.shape, y_train.shape, y_test.shape)

mlr = LinearRegression()

mlr.fit(x_train,y_train)

print(mlr.score(x_test,y_test))


y_pred = mlr.predict(x_test)
print(confusion_matrix(y_test,y_pred))