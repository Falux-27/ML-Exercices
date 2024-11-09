import numpy as np
import matplotlib.pyplot as plt
from fcmeans import FCM

# Générer des données aléatoires
data = np.array([[1, 2], [2, 1], [5, 4], [6, 5]])
print("Donnees:","\n",data,"\n")

# Nombre de clusters
num_clusters = 2
fcm = FCM(n_clusters=num_clusters)
fcm.fit(data)

# Récupérer les centres des clusters et les valeurs d'appartenance
centroids = fcm.centers
print("Les coordonnées des centroïdes:\n", centroids,"\n")
U = fcm.u  # Matrice des degrés d'appartenance
print(" Matrice des degrés d'appartenance:,\n",U)
# Afficher les résultats
plt.scatter(data[:, 0], data[:, 1], c='gray', label='Points de Données', alpha=0.5)
for i in range(num_clusters):
    plt.scatter(centroids[i, 0], centroids[i, 1], marker='x', s=100, label=f'Centroïde {i+1}')
plt.title('Clustering Fuzzy C-Means')
plt.legend()
plt.show()
