import pandas as pd
import numpy as np
from sklearn import linear_model

train = pd.read_csv('student_scores - student_scores.csv')
x = train['Hours']
y = train['Scores']

x=np.expand_dims(x, axis=1)
y=np.expand_dims(y, axis=1)
print('Dataset: ')
print(train)
model = linear_model.LinearRegression()
model.fit(x,y)

y_predict = model.predict(x)
print('score: ',model.score(x,y)) #to get the accuracy of our model

input = float(input('enter a value to predict: ')) #to test any new value as an input
x_test = np.array([input])
x_test = np.expand_dims(x_test, axis=1)
y_test_predict = model.predict(x_test)
print('predicted score is:',y_test_predict)