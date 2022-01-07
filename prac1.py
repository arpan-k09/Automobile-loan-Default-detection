'''/Users/arpankorat/PycharmProjects/pandas-prac/Pandas-Data-Science-Tasks-master/SalesAnalysis/Sales_Data/Sales_August_2019.csv'''

import pandas as pd
import os

# x = pd.read_csv('./Pandas-Data-Science-Tasks-master/SalesAnalysis/Sales_Data/Sales_August_2019.csv')

# Step 1 to merge all csv files
files = [file for file in os.listdir('./Pandas-Data-Science-Tasks-master/SalesAnalysis/Sales_Data')]

# print(files)
allfiles = pd.DataFrame()
for i in files:
    df = pd.read_csv('./Pandas-Data-Science-Tasks-master/SalesAnalysis/Sales_Data/'+i)
    # print(df.shape)
    allfiles = pd.concat([allfiles,df])

# print(allfiles.shape)
allfiles.to_csv('allsalesdata.csv',header=True,index=False)


# Step 2
df = pd.read_csv('allsalesdata.csv')
print(df.head())
# print(df.isna().sum().sort_values(ascending=False).head(15))
# print(df.shape)

nan_rows  = df[df.isna().any(axis=1)]
# print(nan_rows)
df = df.dropna()
# print(df.shape)

df['Months'] = df['Order Date'].str[:2]
# print(df.shape)
# temp = df[df['Months'] == 'Or']
# print(temp.shape)
df = df[df['Months'] != 'Or']
# print(df.shape)
df['Months'] = df['Months'].astype('int32')



df['Price Each'] = df['Price Each'].astype('float32')
df['Quantity Ordered'] = df['Quantity Ordered'].astype('int32')

df['sales'] = df['Quantity Ordered'] * df['Price Each']
sales = df.groupby('Months').sum()['sales']

# print(type(sales))
# print(df['sales'])

