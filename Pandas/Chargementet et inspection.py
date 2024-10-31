
import pandas as pd
import numpy as np 

#Afficher le nombre de valeur unique de chaque colonne
datafrm = pd.DataFrame({
    'Nom': ['Alice', 'Bob', 'Alice', 'Charlie', 'Alice', 'Charlie','Alice'],
    'Âge': [25, 30, 25, 35, 25,54,32],
    'Note': [85, 90, 85, 88, 85,50,42]
})
Uniq_vals = datafrm.nunique()  
print("Les valeurs unique de chaque colonne:\n",Uniq_vals)

#Compter le nombre de valeurs de chaque occurrence  dans une colonne
valucounts = datafrm["Nom"].value_counts()
print("Le nombre de valeurs de chaque occurrence dans la colonne Nom:\n",valucounts,"\n")

#Le nombre de valeurs manquantes:
data = pd.read_csv("data.csv")
print("Le nombre de valeurs manquantes:\n",data["Calories"].value_counts(dropna=False),"\n")
 
  
  
print("<------------------->Les Index<--------------------->")

#Choisir la colonne d'index:
data = pd.read_csv("data.csv", index_col="Duration")
print("\nL'index est la colonne Duration:\n",data.head(),"\n")

#Réinitialiser l'index
index_reset = data.reset_index()
print("L'index n'est plus la colonne Duration:\n",index_reset.head(),"\n")

#Générer l'index
index_generate = pd.RangeIndex(start=0, stop=len(data), step=1)  #Creation de l'index et la plage
data.index = index_generate
print("L'index est maintenant généré:\n",data.head(),"\n")

#La fonction set_index
data = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5],
    'Nom': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Âge': [25, 30, 35, 40, 45],
    'Note': [85, 90, 88, 92, 87]
})
new_data = data.set_index('ID') #Définir la colonne index
print("Définir l'index avec Set_Index :\n",new_data.head(),"\n")

