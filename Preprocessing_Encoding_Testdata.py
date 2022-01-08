import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler


df = pd.read_csv('Test_Dataset.csv')
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
    mean_Val = df[i].median()
    df[i].fillna(value=mean_Val,inplace=True)

df.dropna(inplace=True)


# LabelEncoding ================================================================

label_enc = LabelEncoder()
obj_str = ['Client_Income_Type','Client_Education','Loan_Contract_Type','Type_Organization']

# Actual label encoding on string columns ======================================
for i in obj_str:
    df[i].apply(str)
    df[i] = label_enc.fit_transform(df[i])

# Fill NAs with the mean value of entire column=================================
obj_float = ['Age_Days','Employed_Days','Registration_Days']
for i in obj_float:
    df[i] = df[i].apply(replaceMean)
    mean_Val = df[i].median()
    df[i].fillna(value=mean_Val,inplace=True)

df.to_csv('Clean_Test.csv')



# Solve the data imbalance issue by oversampling ===============================
print('Percentage of the minority class: - ',df.Default.sum()/df.shape[0])

ros = RandomOverSampler(sampling_strategy=0.65,random_state=42)

X_res, y_res = ros.fit_resample(df.drop(columns= ['Default']), df['Default'])

print('Percentage of the minority class: - ',y_res.sum()/y_res.shape[0])
print(y_res.value_counts())

X_res['Default'] = y_res
print(X_res.columns)

X_res.to_csv('Clean_Test.csv')





