import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Génération des données synthétiques
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=30, cmap='viridis')
plt.title("Données générées")
plt.show()

# Initialisation et entraînement du modèle GMM
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)
gmm.fit(X)

# Prédiction des clusters
labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=30)
plt.title("Clusters prédits par le GMM")
plt.show()

# Affichage des probabilités d'appartenance
probs = gmm.predict_proba(X).round()
print("Probabilités d'appartenance des 5 premiers points aux clusters :")
print(probs[:5])
