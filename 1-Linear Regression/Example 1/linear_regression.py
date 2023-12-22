#goal is to fit a straight line through the data

#create an instance of the model, fit (train the model) and then make predictions
#on bunch of unknown values

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


df = pd.read_csv('cells.csv')
print(df)
plt.xlabel('time')
plt.ylabel('cells')
plt.scatter(df.time, df.cells, color = 'red', marker = '+')


#x independent (time) , there can be multiple independent var, that's why we have used dataframe
#y dependent on x - number of cells grow with time, we need to predict y

x_df = df.drop ('cells', axis = 'columns')
y_df = df.cells

print(x_df)
print(x_df.dtypes)
#another way is 
# x_df = x_df[['time']]  #double bracket is imp, if you use single, the datatype won't be object which will cause issue later on!
#the above line of code will help you select columns you want 

reg = linear_model.LinearRegression() #created instance of model
reg.fit(x_df, y_df) #training the model

print("MSE Value: ", reg.score(x_df, y_df))

print('Predicted number of cells = ', reg.predict([[2.3]]))


#y = mx + C

c = reg.intercept_
m = reg.coef_

print('from manual calculation, cells = ', (m*2.3 + c))




#now lets make prediction on data that contains only time, we will predict cells

cells_predict_df = pd.read_csv('cells_predict.csv')
cells_predict_df

predicted_cells = reg.predict(cells_predict_df)
print("Predicted Cells: ", predicted_cells)

cells_predict_df [ 'cells'] = predicted_cells
cells_predict_df


cells_predict_df.to_csv("updated_csv_with_cells.csv")