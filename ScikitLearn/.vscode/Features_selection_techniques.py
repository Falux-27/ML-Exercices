import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

datset = pd.read_csv("/Users/apple/Desktop/ML_Algorithms/Dataset/titanic.csv")
print(datset.isnull().sum())
print(datset.info())

"""Rapport des valeurs manquantes"""
#Convertissons en pourcentage
missing_data = (datset.isnull().sum() / len(datset)) * 100
#Fixons le seuil à 0%
variables = []
for x in datset.columns:
    if missing_data[x]<=0.1:
        variables.append(x)
print(variables)
#Création d'une nouvelle dataset avec les valeurs des variables 
#sans valeurs nulles
new_dataset = datset[variables]
print(new_dataset.isnull().sum())
print(new_dataset)


#Filtre à faible variance

""""Il s'agit d'une technique qui cherche à supprimer des features ayant une variance tres
faible par rapport au seuil de variance que nous fixons arbritairement"""

data =pd.DataFrame({
    'Feature1': [
        5, 10, 300, 230, 25, 30, 305, 410, 425, 530,
        515, 60, 65, 70, 75, 80, 85, 90, 195, 100,
        105, 110, 315, 120, 225, 130, 235, 140, 145, 150
    ],
    'Feature2': [
        12, 23, 34, 45, 56, 67, 78, 89, 90, 101,
        112, 123, 134, 145, 156, 167, 178, 18.9, 19.0, 20.1,
        22, 223, 234, 45, 56, 56, 68, 290, 90, 30
    ],
      'Feature3': [
       4.8, 5, 4, 4, 4.9, 4, 4, 5, 4, 4,
        4.5, 4, 5, 4, 5, 5, 4, 4, 4.8, 5,
        5, 4, 5, 4, 4, 4, 5, 4, 4.2, 4.9
    ]
})
print(data.isnull().sum())
print(data.info())
print(data.head(30),"\n\n")

#Normalisation des données
data_normalized = normalize(data)  #Il s'agit de mettre tous les  valeur dans la meme plage
data_normalized = pd.DataFrame(data_normalized, columns=data.columns)
print(data_normalized.head(30),"\n\n")
#Calculons la variances des données
variance=data_normalized.var()
print(variance)
#Fixons le seuil à 0.006 pour la variance
variables = []
treshold =0.006
for x in data_normalized.columns:
    if variance[x]>= treshold:
        variables.append(x)
print(variables,"\n")
new_data = data[variables]
print(new_data.head(30))