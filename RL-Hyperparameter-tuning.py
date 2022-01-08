import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


def RL(filename):
    df = pd.read_csv(filename)

    Y = df['Default']
    # print(df['Default'])
    X = df.drop(columns= ['Default'])

    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.25)
    # print(x_train.shape,x_test.shape, y_train.shape, y_test.shape)

    rf = RandomForestClassifier(max_features=10,n_estimators=1000)

    rf.fit(x_train,y_train)
    # # print(rf.score(x_train,y_train))
    print(rf.score(x_test,y_test))

    y_pred = rf.predict(x_test)

    print(confusion_matrix(y_test,y_pred))

print('Normally Oversampled : - ',RL('Clean_Train_Oversampling.csv'))
print('SMOTE Oversampled : - ',RL('Clean_Train_SMOTE.csv'))

# ======================================================================================================================
# Hyperparameter tuning
# ======================================================================================================================

# max_featu = np.arange(10,11)
# n_esti = np.arange(990,1010,10)
#
# param_grid = dict(max_features=max_featu,n_estimators=n_esti)
# print(param_grid)
# rf = RandomForestClassifier()
#
# grid = GridSearchCV(estimator=rf,param_grid=param_grid,cv=5)
#
# grid.fit(x_train,y_train)
#
# print(grid.score(x_test,y_test))
