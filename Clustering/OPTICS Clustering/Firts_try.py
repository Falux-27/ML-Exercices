import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.datasets import make_blobs
from matplotlib.colors import ListedColormap

# 1. Générer des données avec des clusters
X, _ = make_blobs(n_samples=100, centers=3, cluster_std=0.60, random_state=0)

# 2. Appliquer l'algorithme OPTICS
# min_samples : nombre de points minimum dans un voisinage pour qu'un point soit un noyau
# xi : seuil de densité minimale pour marquer une transition entre les clusters
# min_cluster_size : taille minimale d'un cluster
optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.1)
optics.fit(X)

# 3. Extraire l'ordre des points et les distances de portée
reachability = optics.reachability_[optics.ordering_]  # Distance de portée (reachability distance)
ordering = optics.ordering_  # Ordre des points explorés

# 4. Créer un Reachability Plot avec des couleurs distinctes pour chaque cluster
plt.figure(figsize=(10, 6))
plt.plot(range(len(X)), reachability, color='b', marker='o', linestyle='-', markersize=5)
plt.title('Reachability Plot (OPTICS)', fontsize=16)
plt.xlabel('Index des points dans l\'ordre OPTICS', fontsize=12)
plt.ylabel('Distance de portée (Reachability distance)', fontsize=12)

# Appliquer une colorbar avec des couleurs distinctes pour chaque cluster
cmap = ListedColormap(['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6'])
plt.scatter(range(len(X)), reachability, c=optics.labels_, cmap=cmap, marker='o', s=50, label="Clusters")
plt.show()

# 5. Affichage des clusters trouvés par OPTICS sur les données en 2D
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=optics.labels_, cmap=cmap, s=50, edgecolors='k')
plt.title('Données avec clustering OPTICS', fontsize=16)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12) 
plt.show()
