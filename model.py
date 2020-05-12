import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import joblib
import pickle
import json


info = ""
#data = pd.read_csv("/content/sample_data/Ruthenium_Prices.csv")
#data.head()

df = pd.read_csv("C:/Users/Youssef Abuelleil/Desktop/Intern/Ruthenium_Prices.csv")
output_df = pd.DataFrame(columns = ['date', 'price'])
#output_df = output_df.append()

counter = 0
for index, row in df.iterrows():
  values = row[0].split(';')
  date = float(values[0].replace("-", ""))
  price =  float(values[1])
  output_df.loc[counter] = [date, price]
  counter+=1

output_df.to_csv('test.csv')

data = pd.read_csv("C:/Users/Youssef Abuelleil/Desktop/Intern/test.csv")

data.head()
data = data.drop(['Unnamed: 0'], axis = 1)


plt.scatter(data.date,data.price)
plt.title('price vs time')

reg = LinearRegression()
labels = data['price']

date = data['date']

labels = np.array(labels).reshape(-1,1)

date = np.array(date).reshape(-1,1)

x_train , x_test , y_train , y_test = train_test_split(date, labels ,test_size = 0.10,random_state =2)

reg.fit(x_train,y_train)
accuracy = reg.score(x_test,y_test)

clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,
          learning_rate = 0.1, loss = 'ls')
clf.fit(x_train, y_train)
print(clf.score(x_test,y_test))

Input=eval(input("please write date format in year/month/day with no space or character between numbers ex: 20200531 such that 2020/05/31"))
Input = float(Input)
z= np.array([[Input]])
y_pred = clf.predict(z)
print(y_pred)