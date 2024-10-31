import pandas as pd
import numpy as np
#Selection de colonne :

   #Par leur noms:
dtabase = pd.DataFrame({
    "cities_names":["Prague",'London','Maryland','Seoul','Johannesburg'],
    "population":[2000000,3000000,25000000,3000000,4000000],
    "Superficie":["138km","12km","67km","55km","74km"]
})
print(dtabase)
list_index =np.array(["Ville1","Ville2","Ville3","Ville4","Ville5"])
dtabase.index=list_index
print("Affichage des villes:\n",dtabase["cities_names"])
print("Affichage de la population:\n",dtabase["population"])

    #Par les méthodes :
        #Avec Loc[]:
nom = dtabase.loc["Ville3"]  #Nom index
print("Les infos de cette villes :\n",nom,"\n")
 
        #Avec Iloc[]:
"""Cette méthode n'est utilisable que lorsque les index du Dataframe sont des entiers"""

inf = dtabase.iloc[0:2] #selection des lignes 1 et 2
inf2 = dtabase.iloc[[0,1]]#Alternative
print(inf)
print(inf2)

