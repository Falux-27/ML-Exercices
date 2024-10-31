import pandas as pd

            #Méthode add():
data = pd.DataFrame({
    "points": [120,345,198,30],
    "total":[654,148,498,348]
})
new_data = data.add(2) #Ajoute ce nombre a tous les valeurs du dataframe
print("Après l'ajout de 2:","\n",new_data)    

            #Méthode aggregate():
données= pd.DataFrame({
  "x": [50, 40, 30],
  "y": [300, 12, 42]
})
print(données,"\n")
agg =données.agg([sum], axis=1) #applique la fonction sum sur chaque ligne
print(agg)


print("  ")
        
            #Méthode apply():
def calcul(a):
    return a.sum()

datagramme =pd.DataFrame({
    "x":[50, 40, 30],
  "y": [300, 12, 42]
})
som = datagramme.apply(calcul)

print("Somme des valeurs :",som)
print("")


            #Méthode astype:
slz= pd.DataFrame({
  "Duration": [50, 40, 45],
  "Pulse": [109, 117, 110],
  "Calories": [409.1, 479.5, 340.8]
})
convert = slz.round().astype("Int64") #On arrondis avant de convertir les valeurs
print("Données après conversion :\n" ,convert)
print("")

            #Méthode at:
#Syntaxe: dataframe.at[index(ligne), étiquette(colonne)]

Names = pd.DataFrame({
    "firstname": ["Sally", "Mary", "John"],
  "age": [50, 40, 30],
  "qualifié": [True, False, False]
})
print(Names.at[0,"firstname"],"\n")


            #Méthode combine:
"""Compare deux datagrammes"""
tab1 = pd.DataFrame({
    "a":[32,12,5,62],
    "z":[26,43,60,55]
})
tab2 = pd.DataFrame({
    "a":[22,17,53,2],
    "z":[26,35,80,5]
})
def comparaison (x,y):
    if (x.sum()> y.sum()):
        return x
    else:
        return y
print("Le tableau avec la plus grande somme est:\n",tab1.combine(tab2,comparaison))

print(" ")
            #Méthode Corr():
"""Recherchez la corrélation (relation) entre chaque colonne du DataFrame"""
Spearman = pd.DataFrame({
  "Duration": [560, 4.0, 45],
  "Pulse": [10, 11.7, 110],
  "Calories": [409.1, 479.5, 340.8]
})
print("La correlation entre chaque colonne:\n",Spearman.corr())


            #Méthode Cov:
"""Elle trouve la covariance de chaque colonne"""
ab1 = pd.DataFrame({
    "a":[32,12,5,62],
    "z":[26,43,60,55]
})
print("La covariance de chaque colonne est :\n",ab1.cov())
print("")

            #Méthode set_axis:
animals = pd.DataFrame({
    "names":['Lion','jaguar','eagles','Elephant'],
    "Weights":['130kg','96kg','12kg','250kg']
})
newdf = animals.set_axis(['n˚1','n˚2','n˚3','n˚4'])
print(newdf,"\n")

            
            #Méthode transpose:
"""La méthode transforme les colonnes en lignes et les lignes en colonnes"""
animals = pd.DataFrame({
    "names":['Lion','jaguar','eagles','Elephant'],
    "Weights":['130kg','96kg','12kg','250kg']
})
print("Avant transposition:\n",animals,"\n")
#transposition:
T = animals.transpose()
print("Après transposition:\n",T)


            #Methode Applymap :
"""Elle applique une fonction à chaque élément d’une dataframe"""  
dtfrm = pd.DataFrame({
  "x": [50, 40, 30],
  "y": [300, 12, 42]
})
def add_one(x):
    return x + 1
data = dtfrm.applymap (add_one)
print("Après l'application de la fonction add_one:\n",data)


      #Méthode merge():
df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'],
                    'value1': [1, 2, 3, 4]})

df2 = pd.DataFrame({'key': ['B', 'D', 'E', 'F'],
                    'value2': [5, 6, 7, 8]})

# Fusion des DataFrames sur la colonne 'key'
merged_df = pd.merge(df1, df2, on='key', how='inner')

print(merged_df) 


  