import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

#load data
dataset = pd.read_csv('/Users/apple/Desktop/ML_Algorithms/Seaborn/stocks.csv', parse_dates=['Date'])

    #Tracé linéaire simple
sns.lineplot(data=dataset, x='Date', y='Volume')

plt.show()

    #Tracé avec des paramètres supplémentaires
"""Seaborn trace trois lignes en fonction des valeurs unique de la colonne 'Name'
en utilisant l'argument hue = """

sns.lineplot(data=dataset, x = "Date",y = "Volume",hue='Name')

plt.show()

    #Personnalisation des lignes
        #La methode palette = 
jeux = pd.read_csv("/Users/apple/Desktop/REGRESSION LOGISTIQUE/social_network_Ads.csv")
sns.set_style('dark')
sns.lineplot(data=jeux, x ='Age',
             y = "EstimatedSalary", 
             hue='Gender', 
             palette=['red','green'],
             style='Gender')

plt.show()

#Personnalisation des tracés linéaires
donnees = pd.read_csv("dataset.csv")
sns.set_style('darkgrid')
#Create plot
fig = sns.lineplot(data=donnees,
                   y='SkinThickness',
                   x='BloodPressure',
                   hue='Outcome',
                   style='Outcome')

#Set title and label
plt.title('Blood Pressure vs Glucose')
plt.xlabel('BloodPressure')
plt.ylabel('SkinThickness')

plt.show()


