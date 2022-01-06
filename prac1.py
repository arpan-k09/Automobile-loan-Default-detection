'''/Users/arpankorat/PycharmProjects/pandas-prac/Pandas-Data-Science-Tasks-master/SalesAnalysis/Sales_Data/Sales_August_2019.csv'''

import pandas as pd
import os

# x = pd.read_csv('./Pandas-Data-Science-Tasks-master/SalesAnalysis/Sales_Data/Sales_August_2019.csv')

#
files = [file for file in os.listdir('./Pandas-Data-Science-Tasks-master/SalesAnalysis/Sales_Data')]

# print(files)
allfiles = pd.DataFrame()
for i in files:
    df = pd.read_csv('./Pandas-Data-Science-Tasks-master/SalesAnalysis/Sales_Data/'+i)
    print(df.shape)
    allfiles = pd.concat([allfiles,df])

print(allfiles.shape)
allfiles.to_csv('allsalesdata.csv',header=True,index=False)
