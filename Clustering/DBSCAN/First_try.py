# Importation des bibliothèques nécessaires
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# Génération d'un ensemble de données en forme de lune
X, _ = make_moons(n_samples=300, noise=0.05)

# Visualisation des points de données générés
plt.scatter(X[:, 0], X[:, 1])
plt.title("Data points")
plt.show()

# Application de l'algorithme DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
dbscan.fit(X)

# Récupération des étiquettes de cluster
labels = dbscan.labels_
print(labels)

# Visualisation des clusters identifiés
unique_labels = set(labels)  # Récupération des étiquettes uniques
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))  # Palette de couleurs

# Boucle pour tracer chaque cluster
for k in unique_labels:
    class_member_mask = (labels == k)  # Masque pour les membres du cluster k
    xy = X[class_member_mask]  # Points du cluster k
    plt.scatter(xy[:, 0], xy[:, 1], color=colors[k], label=k)  # Tracé des point
print(" ")
print(class_member_mask)

plt.title('DBSCAN Clustering')  # Titre du graphique
plt.legend()  # Affichage de la légende
plt.show()  # Affichage du graphique final




