import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

dataset = pd.read_csv("bottle.csv")
dataset=dataset[["Salnty","T_degC"]]
dataset = dataset[:500]
dataset=dataset.dropna(axis=0)
dataset.reset_index(drop=True,inplace=True)

x_label=np.array(dataset['Salnty']).reshape(493,1)
y_label=np.array(dataset['T_degC']).reshape(493,1)
x_train, x_test, y_train, y_test = train_test_split(x_label, y_label, test_size = 0.2, random_state = 100)
regression_model=LinearRegression()
regression_model.fit(x_train,y_train)

plt.figure(figsize=(12,10))
plt.scatter(x_label, y_label,  color='aqua')
plt.plot(x_train, regression_model.predict(x_train),linewidth="4")
plt.xlabel("Temperature",fontsize=22)
plt.ylabel("Salinity",fontsize=22)
plt.title("Linear Regression",fontsize=22)

joblib.dump(regression_model, 'regression_model.pkl')
print('Model dumped')
regression_model = joblib.load('regression_model.pkl')
regression_model_columns = list(x_train)
joblib.dump(regression_model_columns, 'regression_model_columns.pkl')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

dataset = pd.read_csv("bottle.csv")
dataset=dataset[["Salnty","T_degC"]]
dataset = dataset[:500]
dataset=dataset.dropna(axis=0)
dataset.reset_index(drop=True,inplace=True)

x_label=np.array(dataset['Salnty']).reshape(493,1)
y_label=np.array(dataset['T_degC']).reshape(493,1)
x_train, x_test, y_train, y_test = train_test_split(x_label, y_label, test_size = 0.2, random_state = 100)
regression_model=LinearRegression()
regression_model.fit(x_train,y_train)

joblib.dump(regression_model, 'regression_model.pkl')
print('Model dumped')
regression_model = joblib.load('regression_model.pkl')
regression_model_columns = list(x_train)
joblib.dump(regression_model_columns, 'regression_model_columns.pkl')