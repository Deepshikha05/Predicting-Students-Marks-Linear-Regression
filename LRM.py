from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

#path = "C:\\Users\\DEEPSHIKHA SETHI\\Desktop\\Linear Regression with multiple variable\\StudentData.csv"
path = '/home/iso-2/Desktop/StudentData.csv'
data2 = pd.read_csv(path, header=None, names=['State', 'SchoolLocation', 'MotherTongue', 'Gender', 'ParentIncomePerAnnum', 'PercentInClass12', 'AvgNumberofHrsStudied', 'IITSelection'])
print(data2)

state = data2['State'].values
school_loc = data2['SchoolLocation'].values
mother_tongue = data2['MotherTongue'].values
gender = data2['Gender']
parent_income = data2['ParentIncomePerAnnum'].values
percentage = data2['PercentInClass12'].values
num_hrs = data2['AvgNumberofHrsStudied'].values
selection = data2['IITSelection'].values

print("type of percentage")
print(type(percentage))
print(percentage.shape)

parent_income = np.array(parent_income, dtype=float)
percentage = np.array(percentage, dtype=float)
num_hrs = np.array(num_hrs, dtype=float)
selection = np.array(selection, dtype=float)

np.reshape(parent_income, 150)
np.reshape(percentage, 150)
np.reshape(num_hrs, 150)
np.reshape(selection, 150)

print(num_hrs.shape)
print(percentage.shape)
x = np.array([percentage, num_hrs]).T
y = np.array([selection])

reg = LinearRegression()
reg.fit(x,y)
