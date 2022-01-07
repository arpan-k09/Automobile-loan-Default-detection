import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('Train_Dataset.csv')
print(df.shape)
# df.isna().sum()
x = list(df)
# print(x)
na = df.isna()
# print(na)
count_na = {}
for i in x:
    count_na[i] = na[i].sum()
count_na = {k: (v/(df.shape[0]))*100 for k, v in sorted(count_na.items(), key=lambda item: item[1],reverse=True)}
print(count_na)


count = 0
for i,j in count_na.items():
    if count>=6:
        pass
    else:
        df.drop([i],axis=1,inplace=True)
        count+=1

df.drop(["ID_Days", "Child_Count", "Client_Housing_Type", "Accompany_Client", "Client_Marital_Status",
         "Client_Gender", "Phone_Change", "Client_Family_Members", "Application_Process_Day",
         "Application_Process_Hour", "Homephone_Tag", "Mobile_Tag", "Client_Permanent_Match_Tag",
         "Client_Contact_Work_Tag"], axis=1, inplace=True)



isAvg=["Client_Income","Credit_Amount","Loan_Annuity","Population_Region_Relative","Cleint_City_Rating","Score_Source_2"]

def replaceMean(x):
    try:
        return float(x)
    except:
        return np.NAN

for i in isAvg:
    df[i] = df[i].apply(replaceMean)
    mean_Val = df[i].mean()
    df[i].fillna(value=mean_Val,inplace=True)

df.dropna(inplace=True)

label_enc = LabelEncoder()
obj_str = ['Client_Income_Type','Client_Education','Loan_Contract_Type','Type_Organization']
for i in obj_str:
    df[i].apply(str)
    df[i] = label_enc.fit_transform(df[i])

obj_float = ['Age_Days','Employed_Days','Registration_Days']
for i in obj_float:
    df[i] = df[i].apply(replaceMean)
    mean_Val = df[i].mean()
    df[i].fillna(value=mean_Val,inplace=True)

df.to_csv('Clean_Train.csv')
