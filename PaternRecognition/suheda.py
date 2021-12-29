import pandas as pd
import numpy as np 
import os
from sys import exit
import seaborn as sb
import matplotlib.pyplot as plt

menu_options = {
    1: 'Option 1 : KNN',
    2: 'Option 2 : Naive Bayes',
    3: 'Option 3 : Linear Regression',
    4: 'Option 4 : PCA',
    5: 'Exit',
}

def print_menu():
    for key in menu_options.keys():
        print (key, '--', menu_options[key] )

def knn_():
    
    print('Handle option \'Option 1 : KNN\n')
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
    

def naivebayes_(p_a, p_b_given_a, p_not_b_given_not_a):
    print('Handle option \'Option 2 : Naive Bayes\n')
    	# calculate P(not A)
    not_a = 1 - p_a
	# calculate P(B|not A)
    p_b_given_not_a = 1 - p_not_b_given_not_a
	# calculate P(B)
    p_b = p_b_given_a * p_a + p_b_given_not_a * not_a
	# calculate P(A|B)
    p_a_given_b = (p_b_given_a * p_a) / p_b
    return p_a_given_b
 
def linearreg_():
    
    print('Handle option \'Option 3 : Linear Regression\n')
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

def pca_(X , num_components):
    print('Handle option \'Option 4 : PCA\n')
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


if __name__=='__main__':
    
    while(True):
        
        print_menu()
        option = ''
        
        try:
            option = int(input('Enter your choice: '))
        except:
            print('Wrong input. Please enter a number ...')
        #Check what choice was entered and act accordingly
        if option == 1:
            knn_()
            
        elif option == 2:
            #P(Detected) = P(Detected|Spam) * P(Spam) + P(Detected|not Spam) * P(not Spam)
            #P(B) = P(B|A) * P(A) + P(B|not A) * P(not A)
            p_a =float(input("P(A|B)--> P(A) giriniz(yüzde):"))/100 
            print(p_a)
            # P(B|A)
            p_b_given_a = float(input("P(B|A) giriniz(yüzde):"))/100 
            print(p_b_given_a)
            # P(B|not A)
            p_b_given_not_a = float(input("P(B|not A) giriniz (yüzde):"))/100 
            print(p_b_given_not_a)
            # calculate P(A|B)
            result = naivebayes_(p_a, p_b_given_a, p_b_given_not_a)
            # summarize
            print('Tespit edilenlerin spam olma olasılığı --> P(A|B) = %.3f%%' % (result * 100))
           
            
        elif option == 3:           
            linearreg_()
                        
        elif option == 4:
            
            smallnumber = float(input("Enter small number for your random dataframe range: "))
            largenumber = float(input("Enter large number for your random dataframe range: "))
            n = int(input("How many lines should be shown (HINT: Please enter numbers in multiples of 2): "))
            
            x = np.random.randint(smallnumber,largenumber,n).reshape(int(n/2),2) 
            target = np.random.randint(2, size=int(n/2))
            
            mat_reduced = pca_(x, 2)
            print(mat_reduced)
            df = pd.DataFrame(mat_reduced , columns = ['PC1','PC2'])
            
            #df['Target'] = target
            
            dataframe_ = pd.concat([df , pd.DataFrame(target, columns = ['Target'])] , axis = 1)
            plt.figure(figsize = (6,6))
            sb.scatterplot(data = dataframe_  , x = 'PC1',y = 'PC2' ,hue='Target' , s = 60 ,palette= 'icefire')            
                 
        elif option == 5:
            print('Good Bye !!!')
            
            exit()
        else:
            print('Invalid option. Please enter a number between 1 and 5.')