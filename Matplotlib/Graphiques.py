import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d

#Création de graphique a barres:
abs = np.array(["Hong-Kong","Chicago","Oslo","Seoul","Istanbul"])
column = np.array([48.185,285.185,117.143,88.482,16.496])
plt.bar(abs,column, width= 0.5)
plt.xlabel("Villes")
plt.ylabel("PIB\hbts en $")
plt.grid()

plt.show()

#Graphe horizontale:
abs = np.array(["Hong-Kong","Chicago","Oslo","Seoul","Istanbul"])
column = np.array([0.949,0.921,0.961,0.925,0.838])
plt.barh(abs,column)
plt.xlabel("IDH")
plt.ylabel("Villes")

plt.show()


#Graphe circulaires:
abs = np.array(["Hong-Kong","Chicago","Oslo","Seoul","Istanbul"])
column = np.array([0.949,0.921,0.961,0.925,0.838])
plt.pie(column, labels= abs)

plt.show()

#Déconnecter les coins:

étiquette = np.array(["Ferrari","Porsche","Volkswagen","Mercedes-Benz"])
CA= np.array([260.93,196.77,96,211.92 ])
deconnect_coins =[0.1,0.1,0.1,0.1]
plt.pie(CA,labels=étiquette,explode=deconnect_coins, shadow=True )

plt.show()

#Légende:

x = np.array(["Apple","Samsung","Huawei","Microsoft"])
CA_1 = np.array([383.93,196.77,96,211.92 ])
deconnect_coins =[0.1,0.1,0.1,0.1]
plt.pie(CA_1 , explode=deconnect_coins, shadow=True)
plt.legend(x, title = "Companies:")

plt.show()

#Graphe 3D:
fig = plt.figure()
axes = plt.axes(projection = '3d')
plt.show()

#Exemple:
fig= plt.figure()
axes = plt.axes(projection = '3d')
x = np.array([12,3,2,5,7,6])
y = np.array([4,5,6,7,8,9])
z = np.array([1,2,3,4,5,6])
axes.plot3D(x,y,z)
plt.show()

#Graphe 3D de surface:
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)  #Créer la grille de coordonnées 2D 
Z = np.sin(np.sqrt(x**2 + y**2)) #définir la fonction pour les valeurs de z
fig = plt.figure()
ax = fig.add_subplot(111, projection ='3d')
surf = ax.plot_surface(x, y, Z, cmap= 'cividis')
fig.colorbar(surf)
ax.set_title('Graphe 3D de surface')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

#Graphe Nuage de point 3D #:
np.random.seed(0)  #Fixer la graine aléatoire
n = 100
x = np.random.rand(n)
y = np.random.rand(n)
z = np.random.rand(n)
fig = plt.figure()
axy = fig.add_subplot(111, projection = '3d')
axy.scatter(x, y, z) #creation des nuage de points sur le 3D
axy.set_title('Graphe Nuage de point 3D')
axy.set_xlabel('X')
axy.set_ylabel('Y')
axy.set_zlabel('Z')
plt.show()


