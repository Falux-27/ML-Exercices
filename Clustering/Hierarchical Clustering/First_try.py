import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage

# Génération de données pour l'exemple
X, _ = make_blobs(n_samples=20, centers=3, random_state=42)

# Visualisation initiale des données
plt.scatter(X[:, 0], X[:, 1], c='black', marker='o')
plt.title("Données générées")
plt.show()

# Étape 1 : Clustering Hiérarchique Agglomératif avec sklearn

# Modèle de clustering avec agglomératif
model = AgglomerativeClustering(n_clusters=3, linkage='ward')

# Apprentissage du modèle
model.fit(X)

# Affichage des labels de clusters pour chaque point
print("Labels des clusters attribués aux points :", model.labels_)

# Visualisation des clusters
plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap='viridis', marker='o')
plt.title("Clustering Hiérarchique Agglomératif (Résultats)")
plt.show()

# Étape 2 : Création d'un Dendrogramme pour visualiser les fusions des clusters

# Calcul du linkage pour le dendrogramme
Z = linkage(X, method='ward', metric='euclidean')

# Visualisation du dendrogramme
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title("Dendrogramme du Clustering Hiérarchique Agglomératif")
plt.xlabel("Points de données")
plt.ylabel("Distance")
plt.show()

