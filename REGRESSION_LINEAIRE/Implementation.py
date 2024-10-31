import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
print(" ")
print("                            #------->Regression Linéaire simple:--------->")
                            
                        
x = np.array([0,3,4,7,4,9,12,9,6])
y = np.array([2,4,5,8,2,7,10,8,6])

#Calcul du nombre d'observation:
size_x = np.size(x)
print("La taille est de :",size_x,"\n\n")

#Calcul de la moyenne:
moy_x =np.mean(x)
moy_y =np.mean(y)
print("La moyenne des variables indépendantes X est :", moy_x)
print("La moyenne des valeurs réelles est:",moy_y)

#Calcul de la déviation croisée et de la déviation par rapport à x
deviation_XY  =np.sum(x *y)  - size_x*moy_x*moy_y
deviation_XX = np.sum(x*x) - size_x*moy_x*moy_x

#Calcul des coefficients:
b1 = deviation_XY / deviation_XX
b0 = moy_y - b1*moy_x

print("Les coefficients:",b1,b0)

#Prédiction des valeurs de y:
y_predict = b0 + b1 * x

#Tableau des valeurs predict:
values_predict = pd.DataFrame({
    'x':x,
    'y':y,
    "Y_predict": y_predict
})
print("Tableau des tous les valeurs:\n\n",values_predict)

#Représentation graphique des valeur  s réelles:
plt.scatter(x, y, color = "magenta", marker="o")

#Représentation de la ligne de regression:
plt.plot(x, y_predict, color = "green")
plt.xlabel("Variables X")
plt.ylabel("Valeurs prédites de y")

plt.show()

