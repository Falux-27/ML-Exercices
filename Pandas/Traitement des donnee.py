import pandas as pd
import numpy as np

#Les données manquantes

dataset = pd.read_csv("data.csv")

#Fonction isna()

"""Renvoie True si la valeur est manquante et False si la valeur n'est pas manquante"""
print("Les valeurs manquantes:\n",dataset.isna(),"\n")

#Compter les valeurs manquantes
missing_count = dataset.isna().sum() 
print("Le nombre de valeurs manquantes dans chaque colonne:\n",missing_count,"\n")

# Filtrer les lignes avec des valeurs manquantes
missing_data = dataset[dataset.isna().any(axis=1)]
print("Les lignes avec des valeurs manquantes:\n",missing_data,"\n")
 
 
#Remplir les valeurs manquantes

            #fillna:
#remplir avec une valeur spécifique
dataset["Duration"].fillna(value= 60, inplace=True)
print(dataset.head())

#Remplir avec la moyenne

dataset["Calories"].fillna(value=dataset["Calories"].mean(), inplace=True)
print(dataset.to_string())
 
 
#Supprimer les lignes avec valeurs manquantes

            #dropna():
"""Elle supprime toutes les lignes pour lesquelles une colonne est manquante."""
dataset.dropna(inplace=True)
 
 #Supprimer des lignes des colonnes ou des valeurs manques
dataset.dropna(subset=['Calories','Duration'], how= 'any') 


#Convertir les types de donnees

    #Astype():
    
data = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5],
    'Nom': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Âge': [25, 30, 35, 40, 45],
    'Note': [85, 90, 88, 92, 87]
})
print("Originale Dataframe:\n", data,"\n")
 #Conversion
data['Âge'] = data['Âge'].astype(float)  
print("Dataframe avec l'age convertis en décimale:\n",data,"\n")

#Convertir les colonne en même temps

data = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5],
    'Nom': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Âge': [25, 30, 35, 40, 45],
    'Note': [85, 90, 88, 92, 87]
})

data_convert = data.astype({'Âge':'float', "Note":'float'})    #Conversion
print("conversion simultanée des colonne Age et Note:\n",data_convert,"\n")


#Opérations sur les chaînes

data = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5],
    'Nom': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Degree':['High School','Undergraduate','PhD','Master','Certificate'],
    'Âge': [25, 30, 35, 40, 45],
    'Note': [85, 90, 88, 92, 87]
})
#Convertir une colonne en majuscule
data['Nom']=data['Nom'].str.upper()

#Convertir une colonne en minuscule
data['Degree'] = data['Degree'].str.lower()
print("After conversion en majuscule et minuscule:,\n",data,"\n\n")

#Remplacer une chaîne par une autre

dataset = pd.DataFrame({
     'Nom': ['Alice Smith', 'Bob Brown', 'Charlie Clark', 'David Davis', 'Eva Evans'],
    'Adresse': ['123 Main St.', '456 Maple St.', '789 Oak St.', '101 Pine St.', '202 Birch St.']
})
print("Le dataset original:\n", dataset,"\n")

#La fonction .str.replace()

dataset['Nom'] = dataset['Nom'].str.replace(" ",'-')
dataset['Adresse'] = dataset['Adresse'].str.replace("St.","Street")

print("Le dataset après remplacement:\n",dataset,"\n")


#Grouper les données

    #Concaténation 
data1 = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C':[1,2,3]}
data2 = {'A': [7, 8, 9], 'B': [10, 11, 12], 'D':[1,2,3]}
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

#Concaténation verticale :(colonne)
result = pd.concat([df1,df2],axis=0)
print("Concaténation verticale:\n",result,"\n")

##Concaténation horizontale
data1 = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C':[1,2,3]}
data2 = {'A': [7, 8, 9], 'B': [10, 11, 12], 'D':[1,2,3]}
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
result2 = pd.concat([df1,df2],axis=1)
print("Concaténation horizontale:\n",result2)

 




