import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('Clean_Train.csv')

Y = df['Default']
print(df['Default'])
X = df.drop(columns= ['Default'])

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.25)
print(x_train.shape,x_test.shape, y_train.shape, y_test.shape)

rf = RandomForestClassifier(max_features=10,n_estimators=1000)

rf.fit(x_train,y_train)

print(rf.score(x_test,y_test))