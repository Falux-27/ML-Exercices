import pandas as pd 

#Creation de de Dataframe:
data = {
    "calories": [450,125,237],
    "duration": [54,31,45],
    "Vitamins":['A','B','C']
}
dftrm = pd.DataFrame(data)

    #Affichage:
print(dftrm)


#Localiser une ligne:
    #Loc:
print(dftrm.loc[0])
    #Localiser plusieurs lignes:
print(dftrm.loc[[0,2]])

#Nommés des index:
    #Index:
data =pd.DataFrame ({
    "calories": [450,125,237],
    "duration": [54,31,45],
    "Vitamins":['A','B','C']
})
list_index = ["Jour1","Jour2","Jour3"]
data.index = list_index
print(data,"\n")

#Localiser les index nommés:
print(data.loc["Jour2"],"\n")

#Sélectionner le contenu d'une cellule:

    #Iloc:
       #Syntaxe= dataframe.iloc[ligne, colonne]
       
print("Valeur de la ligne 3 et colonne 1:",data.iloc[2,0])


            