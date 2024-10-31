import pandas as pd 

#Lire les fichiers CSV:
    #To_string:
dtfrm = pd.read_csv("data.csv")
#Remplacer les valeurs vide:

    #Fillna():
dtfrm.fillna(value=130,inplace=True)
print(dtfrm.to_string())

#Méthode 2:
    #Sur une ligne:
dtfrm.loc[7, "Duration"] = 45 #Change le nombre de la ligne 7 de la colonne "Durée" par 45

    #Colonne spécifiée:
dtfrm = pd.read_csv("data.csv")
dtfrm["Duration"].fillna(25, inplace=True) 
dtfrm.dropna(inplace=True)   #Supprimer les lignes vides 
print(dtfrm.to_string())

#Supprimer des lignes :
for x in dtfrm.index:
    if dtfrm.loc[x,"Duration"]<20:
        dtfrm.drop(x)
print(dtfrm.to_string(),"\n")

#Supprimer les doublons:
dtfrm.drop_duplicates(inplace=True)
print(dtfrm.to_string())



