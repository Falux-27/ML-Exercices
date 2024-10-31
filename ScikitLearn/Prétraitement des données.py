import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression 
import pandas as pd 
import matplotlib.pyplot as plt

#Binarization:
"""Cette technique de prétraitement est utilisée lorsque
nous devons convertir nos valeurs numériques en valeurs booléennes.""" 
input = np.array([
    [2.1, -1.9, 5.5],
   [-1.5, 2.4, 3.5],
   [0.5, -7.9, 5.6],
   [5.9, 2.3, -5.8]])
data_binarized = preprocessing.Binarizer(threshold=0.5).transform(input)    #Avec threshold =0.5 qui est le seuil tous les valeur supérieurs a 0.5 seront convertir en 1 
print("############################################# " )                    #et les autres valeurs inférieurs a 0.5 ou égales seront convertis en 0
print("Données binaires:\n",data_binarized,"\n")

#Retrait moyen:

    #preprocessing.scale()
"""Cette technique est utilisée pour éliminer la moyenne du vecteur caractéristique 
en les centrant autour de zéro et en leur donnant une variance d'unité."""

Input_data = np.array([
   [2.1, -1.9, 5.5],
   [-1.5, 2.4, 3.5],
   [0.5, -7.9, 5.6],
   [5.9, 2.3, -5.8]]
)
print(Input_data,"\n")
print("la moyennes de chaque colonne:\n","Mean =",Input_data.mean(axis=0) ,"\n")
print("La variance des valeurs des colonnes :\n","Variance=", Input_data.std(axis=0),"\n")

     #Retrait de la moyenne et l’écart-type:
     
data_normalized = preprocessing.scale(Input_data) 
print(data_normalized)
print("La moyenne centrée:\n",data_normalized.mean(axis=0),"\n")
print("La variance des données : \n",data_normalized.std(axis=0),"\n")


#Mise en échelle:

    #preprocessing.MinMaxScaler()
""" Il met à l'échelle les caractéristiques des données de sorte que leurs
valeurs se situent dans une plage spécifiée,exemple entre 0 et 1."""

data = np.array(
    [
      [2.1, -1.9, 5.5],
      [-1.5, 2.4, 3.5],
      [0.5, -7.9, 5.6],
      [5.9, 2.3, -5.8]
   ]
)
#Creation de l'intervalle
data_scaler= preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaler = data_scaler.fit_transform(data)   #mise en échelle
print("Après mise en échelle des valeurs:\n\n", data_scaler,"\n\n")


#Normalisation:

    #L1 normalisation:
"""C'est une méthode de mise à l'échelle des vecteurs de caractéristiques.
Elle ajuste les valeurs de sorte que la somme des valeurs absolues soit égale à 1."""

xg = np.random.rand(4,3)
print("matrice originale:\n",xg,"\n\n")
normalisation_L1 = preprocessing.normalize(xg, norm= 'l1')
print("Les données normalisées avec la norme de L1:\n\n",normalisation_L1,"\n\n")


#Représentation des données:
from sklearn.datasets import load_iris


#Selection des caractéristiques 
"""Permet de sélectionner les caractéristiques les plus significatives à partir d'un modèle de ML"""
# Charger les données
iris = load_iris()
data = iris.data
target = iris.target

#Instanciation avec la régularisation lasso
model = LogisticRegression(max_iter=10000, solver='liblinear', penalty='l1')

#Selectionner les caracteristique significatives
selector = SelectFromModel(estimator=model).fit(data, target)

#Afficher les caractéristiques sélectionnées
print("Les caractéristiques sélectionnées:\n\n",selector)

