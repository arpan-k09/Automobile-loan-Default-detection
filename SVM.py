from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

svm = SVC()
def SV(filename):
    df = pd.read_csv(filename)

    Y = df['Default']
    # print(df['Default'])
    X = df.drop(columns=['Default'])
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

    svm.fit(x_train,y_train)
    # print(svm.score(x_test,y_test))
    y_pred = svm.predict(x_test)
    print(confusion_matrix(y_test,y_pred))
    tp,fp, fn, tn = confusion_matrix(y_test,y_pred).ravel()
    print((tn+tp)/(tp+fp +fn+ tn))

print('Normally Oversampled : - ',SV('Clean_Train_Oversampling.csv'))
print('SMOTE Oversampled : - ',SV('Clean_Train_SMOTE.csv'))