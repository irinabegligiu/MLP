import numpy as np
from sklearn import neural_network

#read and import the database file into a numpy matrix
data=np.genfromtxt(r'data.txt')
#the database contains 4 columns: data on the first 3 and the tag on the last
#the database has two separate tags

X=data[:,:3] #first 3 collumns in the X matrix
Y=data[:,3] #fourth column in a tag array "Y"

#initializing the training matrices
X_train=np.concatenate((X[:int(0.75*50859)], X[50859:int(0.75*194198)]))
Y_train=np.concatenate((Y[:int(0.75*50859)], Y[50859:int(0.75*194198)]))
#X_train, Y_train = 75% of (first category; 50859 elements) + 
# + 75% of (second category; 194198 elements)

#initializing the testing matrices
X_test=np.concatenate((X[int(0.75*50859):50859], X[int(0.75*194198):194198]))
Y_test=np.concatenate((Y[int(0.75*50859):50859], Y[int(0.75*194198):194198]))
#X_test, Y_test = 25% of (first category; 50859 elements) +
# + 25% of (second category; 194198 elements)

def rata(pred, test): #success rate function
    nr=0
    for i in range (len(X_test)):  #for every value
        if pred[i]==test[i]: #we check if the prediction was correct
            nr+=1 #it counts the correct predictions
    return 100*nr/len(X_test) #return the percentage


#we calculate the average accuracy for training and testing in 15 iterations
suma=0
for i in range(15):
    clf=neural_network.MLPClassifier(hidden_layer_sizes=(10,10), learning_rate_init=0.1, max_iter=1500)
    
    #algorithm train
    clf.fit(X_train,Y_train)
    
    #algorithm test
    predictie=clf.predict(X_test)
    
    #success rate calculator
    suma = suma + rata(predictie, Y_test)
print(str(round(suma/15, 4)) + "%") #return the accuracy percentage with a precision of 4 decimals


