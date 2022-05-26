import numpy as np
from sklearn import neural_network

#citim si importam fisierul cu baza de date intr-o matrice numpy
data=np.genfromtxt(r'data.txt')
#baza de date contine 4 coloane: primele 3 de date si ultima de etichete
#baza de date are doua etichete(categorii) distincte

X=data[:,:3] #Includem in matricea X primele 3 coloane
Y=data[:,3] #Includem coloana 4 intr-un sir de etichete Y

#initializam matricile X de training + test ==================================
X_train=np.concatenate((X[:int(0.75*50859)], X[50859:int(0.75*194198)]))
Y_train=np.concatenate((Y[:int(0.75*50859)], Y[50859:int(0.75*194198)]))
#X_train, Y_train = 75% din (prima categorie; 50859 de elemente) + 
# + 75% din (a doua categorie; 194198 de elemente)

#initializam sirurile Y de training + test ===================================
X_test=np.concatenate((X[int(0.75*50859):50859], X[int(0.75*194198):194198]))
Y_test=np.concatenate((Y[int(0.75*50859):50859], Y[int(0.75*194198):194198]))
#X_test, Y_test = 25% din (prima categorie; 50859 de elemente) +
# + 25% din (a doua categorie; 194198 de elemente)

def rata(pred, test): #functie pentru a calcula rata de succes a clasificarii
    nr=0
    for i in range (len(X_test)):  #pentru fiecare valoare de test
        if pred[i]==test[i]: #verificam daca predictia a fost corecta
            nr+=1 #numaram de cate ori a fost corecta
    return 100*nr/len(X_test) #calculam procentajul (rata de succes)


#calculam media acuratetei pentru antrenari si testari in 15 iteratii
suma=0
for i in range(15):
    clf=neural_network.MLPClassifier(hidden_layer_sizes=(10,10), learning_rate_init=0.1, max_iter=1500)
    
    #antrenam algoritmul  
    clf.fit(X_train,Y_train)
    
    #testam algoritmul
    predictie=clf.predict(X_test)
    
    #calculam rata de succes a clasificarii folosind functia <<rate(pred, test)>> definita anterior
    suma = suma + rata(predictie, Y_test)
print(str(round(suma/15, 4)) + "%") #afisam procentajul acuratetei rotunjit la 4 zecimale


