import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

# Load dataset
dataset = pd.read_csv('Students data.csv')

#-->La fonction relplot()
"""Créer des nuages de point"""

sns.relplot(data=dataset, x = 'Calculus1', y = 'Calculus2')
plt.show()

#Personnalisation des tracés Seaborn 
    
    #--->La fonction set_style()
""""Elle permet de changer la couleur du background"""

sns.set_style('darkgrid')
sns.scatterplot(data=dataset, x = 'Calculus1', y = 'Statistics')
plt.show()

    #--->La paramètre hue = 
"""Permet de colorier les points fonction des valeurs uniques de la colonne catégorielle"""

sns.set_style('dark')
sns.relplot(data=dataset,
            x = 'Calculus1', 
            y = 'Statistics', 
            hue = 'gender')
plt.show()

        #--->Création de plusieurs cartes
"""col = permet de créer des sous-graphiques pour chaque valeur unique de la colonne"""

sns.set_style('darkgrid')
sns.relplot(data=dataset,
            x= 'Probability',
            y= 'Calculus1',
            hue = 'class',
            col='gender',)
plt.show()


   #---->Création de tracés catégoriels
   
    #--->La fonction catplot()
"""Elle crée des graphes de types catégorielle ou les barres représentent une catégorie particulière"""
sns.set_style('dark')
sns.catplot(data=dataset,
            x = 'gender',
            y = 'GPA',
            hue='class',
            kind='bar')
plt.show()