import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics

# Load dataset
match = pd.read_csv('/Users/apple/Desktop/ML_Algorithms/Random Forest/matches.csv')
deliveries = pd.read_csv('/Users/apple/Desktop/ML_Algorithms/Random Forest/deliveries.csv')
print(match.head(5),"\n\n")
print(deliveries.head(5),"\n\n")

#Analyse Exploratoire

liste = ['match','deliveries']
row_match = match.shape[0]
col_match = match.shape[1]
row_deliveries = deliveries.shape[0]
col_deliveries = deliveries.shape[1]
print(f'Il y\'a {row_match} lignes et {col_match} colonnes dans le dataset Match',"\n")
print(f'Il y\'a {row_deliveries} lignes et {col_deliveries} colonnes dans le dataset Deliveries',"\n\n")

#Columns
print('Colonnes Match :',match.columns, "\n")
print('Colonnes Deliveries :',deliveries.columns, "\n\n")

#Infos
print('Match :',match.info(), "\n")
print('Deliveries :',deliveries.info(), "\n\n")

#Colonnes nulles
print('Colonnes Match avec des valeurs nulles :',match.columns[match.isnull().any()].tolist(),"\n")
print('Colonnes Deliveries avec des valeurs nulles :',deliveries.columns[deliveries.isnull().any()].tolist(),"\n\n")

#Nombre de valeur nulle par colonne
print('Nombre de valeurs nulle dans Match :',match.isnull().sum(),"\n")
print('Nombre de valeurs nulle dans Deliveries :',deliveries.isnull().sum(),"\n\n")

#Supprimer les colonnes inutiles
match = match.drop(columns=['id', 'date', 'toss_winner', 'toss_decision', 'winner', 
                    'win_by_runs', 'player_of_match', 'venue',
                    'umpire1', 'umpire2', 'umpire3'])

deliveries = deliveries.drop(columns=['match_id', 'batsman','inning', 'non_striker', 
                     'bowler', 'player_dismissed', 'fielder'])

#Valeurs uniques
liste_match = match.columns.tolist()
for col in liste_match:
     print('Les valeurs unique de la colonne {col} :',match[col].unique(),"\n")
liste_deliveries = deliveries.columns.tolist()
for x in liste_deliveries:
    print('Les valeurs uniques de la colonne {x}:',deliveries[x].unique(),"\n\n")

#Prétraitement des données

totalrun_df=deliveries.groupby(['match_id','inning']).sum()['total_runs'].reset_index()
totalrun_df.columns=['match_id','inning','total_runs_inning']
print(totalrun_df)

 