import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np


#Étiquettes Matplotlib:
dtabase = pd.DataFrame({
    "cities_names":["Prague",'London','Maryland','Seoul','Johannesburg'],
    "population":[2.000,3.000,2.500,3000,4000]
})
plt.plot(dtabase["cities_names"],dtabase["population"])
plt.xlabel("Villes")
plt.ylabel("Population")
plt.show()

#Titre:
economics = pd.DataFrame({
    "country":["Nigeria","South-Africa","Égypte","Algérie","Kenya"],
    "PIB":[514,419,394,193,177]
})
plt.plot(economics["country"],economics["PIB"])
plt.xlabel("Pays")
plt.ylabel("PIB en milliard USD")
plt.title("PIB en milliard $ de 2023-2024 ",loc= 'center')
plt.show()


#Ajout de lignes de grille
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.plot(x,y)
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")
plt.title("Sport Watch Data")
#Grille:
plt.grid()
plt.show()

#Spécifier les lignes de grille:Avec AXIS= 'x' 'both' or 'y'
economics = pd.DataFrame({
    "country":["Nigeria","South-Africa","Égypte","Algérie","Kenya"],
    "PIB":[514,419,394,193,177]
})
plt.plot(economics["country"],economics["PIB"],linewidth = 3.5, marker = "o")
plt.xlabel("Pays")
plt.ylabel("PIB en milliard USD")
plt.title("PIB en milliard $ de 2023-2024 ",loc= 'right')
#Grille:
plt.grid(axis='both',linestyle = "--",color = "black")
plt.show()


