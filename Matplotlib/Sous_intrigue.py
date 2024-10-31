import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

#Afficher plusieurs tracés:

#La fonction Subplot:
"""Elle prends trois paramètres :
La 1er:le nombre de ligne
La 2nd:le nombre de colonne 
La 3em:Le numéro du graphe dans l'ordre"""

#Figure 1:
fig1 =pd.DataFrame({
    "companies":["Apple","Samsung","Huawei","Microsoft"],
    "sales":[8,10,5,9]
})
plt.subplot(2,2,1)
plt.plot(fig1["companies"],fig1["sales"])
plt.title("Vente d'appareil électronique 2023-2024")
plt.xlabel("Entreprises")
plt.ylabel("Nombre de vente en millions ")

#Figure 2:
fig2 = pd.DataFrame({
    "comp":["Apple","Samsung","Huawei","Microsoft"],
    "C.A":[383.93,196.77,96,211.92 ]
})
plt.subplot(2,2,2)
plt.plot(fig2["comp"],fig2["C.A"])
plt.title("Chiffre d'Affaire en milliard USD")
plt.xlabel("Entreprises")
plt.ylabel("Chiffre d'affaire")

plt.suptitle("Bilan annuel")
plt.show()

#Nuage de points :

revenues = pd.DataFrame({
    "companies": ["Ferrari","Porsche","Volkswagen","Mercedes-Benz"],
    "C.A":[5.5,36,17,150]
})
plt.scatter(revenues["companies"],revenues["C.A"])
plt.title("Chiffre d'Affaire en milliard USD")
plt.xlabel("Marques")
plt.ylabel("Chiffre d'Affaire")

plt.suptitle("Bilan annuel")
plt.show()


#Comparer des données:

x = np.array(["Apple","Samsung","Huawei","Microsoft"])
CA_1 = np.array([383.93,196.77,96,211.92 ])
plt.scatter(x, CA_1)
plt.xlabel("CHiffre d'affaire ")
plt.ylabel("Entreprise")
#############################################################
x = np.array(["Apple","Samsung","Huawei","Microsoft"])
CA_2 = np.array([350.105,201.93,102,201.92 ])
plt.scatter(x, CA_1, s = 45)
plt.title("Chiffre d'affaire 2023 et 2024")
plt.xlabel("CHiffre d'affaire ")
plt.ylabel("Entreprise")
plt.scatter(x, CA_2,s = 25)

plt.show()



