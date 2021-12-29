# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 18:33:34 2021

@author: Lenovo
"""

import pandas as pd
import numpy as np
import math
import operator
import matplotlib.pyplot as plt

data = pd.read_csv('headbrain.csv')
print(data.head())

X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values

# calculate mean of x & y using an inbuilt numpy method mean()
mean_x = np.mean(X)
mean_y = np.mean(Y)

b = len(X)

# using the formula to calculate m & c
numer = 0
denom = 0
for i in range(b):
  numer += (X[i] - mean_x) * (Y[i] - mean_y)
  denom += (X[i] - mean_x) ** 2
b = numer / denom
a = mean_y - (b * mean_x)

print (f'b = {b} \na = {a}')

# plotting values and regression line
max_x = np.max(X) + 100
min_x = np.min(Y) - 100

# calculating line values x and y
x = np.linspace (min_x, max_x, 100)
y = a + b * x

plt.plot(x, y, color='#58b970', label='Regression Line')
plt.scatter(X, Y, c='#ef5423', label='data points')

plt.xlabel('Head Size in cm')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()

# calculating R-squared value for measuring goodness of our model. 

ss_t = 0 #total sum of squares
ss_r = 0 #total sum of square of residuals

for i in range(len(x)): # val_count represents the no.of input x values
  y_pred = a + b * X[i]
  ss_t += (Y[i] - mean_y) ** 2
  ss_r += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_r/ss_t)

print(r2)