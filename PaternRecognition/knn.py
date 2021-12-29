# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 15:44:48 2021

@author: Lenovo
"""
import pandas as pd
import numpy as np


df = pd.read_pickle("a_file.pkl")
a=df.iloc[:,1:]

X = a.values.tolist()
Y= df['index'].to_numpy().tolist()


# Find which variable is the most in an array of variables
def most_found(array):
    list_of_words = []
    for i in range(len(array)):
        if array[i] not in list_of_words:
            list_of_words.append(array[i])
            
    most_counted = ''
    n_of_most_counted = None
    
    for i in range(len(list_of_words)):
        counted = array.count(list_of_words[i])
        if n_of_most_counted == None:
            most_counted = list_of_words[i]
            n_of_most_counted = counted
        elif n_of_most_counted < counted:
            most_counted = list_of_words[i]
            n_of_most_counted = counted
        elif n_of_most_counted == counted:
            most_counted = None
            
    return most_counted

def find_neighbors(point, data, labels, k=3):
    # How many dimentions do the space have?
    n_of_dimensions = len(point)
    
    #find nearest neighbors
    neighbors = []
    neighbor_labels = []
    
    for i in range(0, k):
        # To find it in data later, I get its order
        nearest_neighbor_id = None
        smallest_distance = None
        
        for i in range(0, len(data)):
            eucledian_dist = 0
            for d in range(0, n_of_dimensions):
                dist = abs(point[d] - data[i][d])
                eucledian_dist += dist
                
            eucledian_dist = np.sqrt(eucledian_dist)
            
            if smallest_distance == None:
                smallest_distance = eucledian_dist
                nearest_neighbor_id = i
            elif smallest_distance > eucledian_dist:
                smallest_distance = eucledian_dist
                nearest_neighbor_id = i
                
        neighbors.append(data[nearest_neighbor_id])
        neighbor_labels.append(labels[nearest_neighbor_id])
        
        data.remove(data[nearest_neighbor_id])
        labels.remove(labels[nearest_neighbor_id])
    return neighbor_labels

def k_nearest_neighbor(point, data, labels, k=3):
    
    # If two different labels are most found, continue to search for 1 more k
    while True:
        neighbor_labels = find_neighbors(point, data, labels, k=k)
        label = most_found(neighbor_labels)
        if label != None:
            break
        k += 1
        if k >= len(data):
            break
    print(label)         
    return label

print("Yeni pointi giriniz: (x,y)")
point = [ float(x) for x in input().split()]
  
print("array:", point)

k_nearest_neighbor(point, X, Y, k=5)
