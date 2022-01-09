from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

lr = LogisticRegression(max_iter=100000000)
def LR(filename):
    df = pd.read_csv(filename)
    # print(df.shape)
    Y = df['Default']
    # print(df['Default'])
    X = df.drop(columns=['Default'])
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
    # print(x_train.shape)
    lr.fit(x_train,y_train)
    print(lr.score(x_train,y_train))
    y_pred = lr.predict(x_test)
    print(confusion_matrix(y_test,y_pred))
    tp,fp, fn, tn = confusion_matrix(y_test,y_pred).ravel()
    print((tn+tp)/(tp+fp +fn+ tn))

print('Normally Oversampled : - ',LR('Clean_Train_Oversampling.csv'))
print('SMOTE Oversampled : - ',LR('Clean_Train_SMOTE.csv'))