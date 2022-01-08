from dabl import plot
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Clean_Train.csv')

plot(df,'Default')
plt.show()