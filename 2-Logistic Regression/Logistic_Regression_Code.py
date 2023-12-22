import pandas as pd
from matplotlib import pyplot as plt


#STEP 1: Data Reading and Understanding!
df  = pd.read_csv("D:\Anum\Learning ML DL\Logistic Regression\Images_Analyzed_Productivity.csv")
print (df.head())

#understand how the dataset actually looks like, using different plots
plt.scatter(df.Age, df.Productivity, marker = "+", color = 'red')
# we want dataset to be well balanced
sizes = df['Productivity'].value_counts(sort = 1)
plt.pie(sizes, autopct = '%1.1f%%')



#STEP 2: Drop irrelevant data (Data Cleaning)
df.drop(['Images_Analyzed'], axis = 1, inplace  = True)
df.drop(['User'], axis =1, inplace = True)
print(df.head())



#STEP 3: Deal with Missing Values
#this step is usedful when we have missing values in the dataset

#STEP 4: Convert output values Good to 1 and Bad to 0 (non-numeric to numeric)
df.Productivity[df.Productivity == 'Good '] = 1
df.Productivity[df.Productivity == 'Bad'] = 2
print(df.head())


#STEP5: Let's prepare the data (define independent and dependent variables)
#dependent variables
Y = df['Productivity'].values
#we will have to convert array of objects to int
Y= Y.astype('int')

#independent variables
X = df.drop(labels = ['Productivity'], axis = 1)
X.head()


#STEP6: Split data into train and test 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state=20)
#random state acts as seed, and gives same split everytime


#STEP7: Define Machine Learning Model
from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression() #created an instance of the model
logistic_model.fit(X_train, Y_train)

#STEP8: Let's test the model now!
prediction_test = logistic_model.predict(X_test)

#STEP9: Evaluate how good the model is
from sklearn import metrics

print('Accuracy: ', metrics.accuracy_score(Y_test, prediction_test))
#got an accuracy of 71.42%

#we can also look at the weights, to see which factors are impacting the 
#results the most!

#STEP10: Understand weights

print(logistic_model.coef_)
# output is something like this
# [[ 0.06641751  0.01327195 -0.02543334]]

weights  = pd.Series(logistic_model.coef_[0])
print(weights)
#Output:
#0    0.066418
#1    0.013272
#2   -0.025433

#but we need to know specific column that weight is associated to!
weights  = pd.Series(logistic_model.coef_[0], index = X.columns.values)
print(weights)

#Output
#Time      0.066418
#Coffee    0.013272
#Age      -0.025433




