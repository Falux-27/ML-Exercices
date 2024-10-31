import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

# Load dataset
dataset = pd.read_csv('/Users/apple/Desktop/ML_Algorithms/Seaborn/Students data.csv')

#--->Créer un diagramme à barres 

   #--->La fonction Barplot()
"""Elle crée des diagrammes à barres pour une variable catégorielle"""
sns.set_style('darkgrid')
sns.barplot(data=dataset, x = 'y', y = 'GPA')
plt.title('Diagramme à barre')
plt.show()

        #---->Diagramme regroupé
sns.set_style('darkgrid')
sns.barplot(data=dataset, x = 'gender', y = 'GPA', hue='class')
plt.title('Diagramme à barre regroupé')
plt.show()

        #---->Modifier l'estimateur
"""Par défaut, la fonction barplot utilise la moyenne pour chaque catégorie"""

                #-->La méthode  estimator = 
"""Permet de changer l'estimateur à utiliser, par exemple l’écart-type"""
sns.set_style('darkgrid')
sns.barplot(data=dataset, x = 'Calculus1', 
            y = 'Probability', 
            hue='gender',
            estimator='std')
plt.show()

        #---->Diagramme à barres horizontale
               #-->La methode orient =
"""Permet de changer la direction des barres, par défaut horizontal"""
sns.set_style('darkgrid')
sns.barplot(data=dataset, x = 'Statistics', 
            y = 'gender', 
            hue='class',
            orient='h')
plt.show()

        #Supprimer les barres d'erreurs
        
        #-->La methode errorbar =
sns.set_style('darkgrid')
sns.barplot(data=dataset, x = 'Calculus1', 
            y = 'y', 
            hue='gender',
            errorbar=None)
plt.show()
