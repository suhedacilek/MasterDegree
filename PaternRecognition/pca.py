# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 15:00:59 2021

@author: Lenovo
"""

import pandas as pd
import numpy as np 


def PCA(X , num_components):
     
    #Step-1
    X_meaned = X - np.mean(X , axis = 0)
     
    #Step-2
    cov_mat = np.cov(X_meaned , rowvar = False)
     
    #Step-3
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
     
    #Step-4
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
     
    #Step-5
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]
     
    #Step-6
    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
     
    return X_reduced


smallnumber = float(input("Enter small number for your random dataframe range: "))
largenumber = float(input("Enter large number for your random dataframe range: "))
n = int(input("How many lines should be shown (HINT: Please enter numbers in multiples of 2): "))

x = np.random.randint(smallnumber,largenumber,n).reshape(int(n/2),2) 
target = np.random.randint(2, size=int(n/2))

mat_reduced = PCA(x , 2)

df = pd.DataFrame(mat_reduced , columns = ['PC1','PC2'])

#df['Target'] = target

dfdf = pd.concat([df , pd.DataFrame(target, columns = ['Target'])] , axis = 1)

import seaborn as sb
import matplotlib.pyplot as plt
 
plt.figure(figsize = (6,6))
sb.scatterplot(data = dfdf , x = 'PC1',y = 'PC2' ,hue='Target' , s = 60 , palette= 'icefire')