import numpy as np 
import matplotlib.pyplot as  plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression

#Generer les valeurs 
reel =np.random.binomial(1,0.9,size=100)
predicted = np.random.binomial(1, 0.9, size=100)

#Créer la matrice de confusion
matrice = confusion_matrix(reel, predicted)
print("La matrice de confusion:\n",matrice,"\n")

#Visualisation de matrice de confusion
fig = ConfusionMatrixDisplay(confusion_matrix= matrice, display_labels=[0,1])
fig.plot()
plt.show()

print(" ")

            #Accuracy_score:
"""Cette precision mesure la fréquence à laquelle le modele est correct"""
precision = metrics.accuracy_score(reel, predicted)
print("Accurracy-score = ", precision)

            #Recall
"""Recall (parfois appelée rappel) mesure la 
capacité du modèle à prédire les résultats positifs."""
recall = metrics.recall_score(reel, predicted)
print("Recall-score = ", recall)

            #F1-score
"""Le F1-score est la « moyenne harmonique » de la précision et de la sensibilité."""
f1_score = metrics.f1_score(reel, predicted)
print("F1-score = ", f1_score)



print("<-------------------Cross-Validation------------------------>")
from sklearn.datasets import load_iris
iris = load_iris()
x, y = iris.data, iris.target
classifier = LogisticRegression()
fold_numb = 5
k_folds = KFold(n_splits=fold_numb, shuffle=True, random_state=42)
cross_val = cross_val_score(classifier, x,y, cv=k_folds)
print(f'Cross-Validation Results (Accuracy): {cross_val}')
print(f'Mean Accuracy: {cross_val.mean()}')

