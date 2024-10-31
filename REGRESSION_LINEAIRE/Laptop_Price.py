import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score , mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score , KFold
from sklearn.preprocessing import OneHotEncoder,StandardScaler
 

#Chargement des données
dataset = pd.read_csv("laptop_price.csv",encoding='latin1')

#About data
#-->Conversion de la colonne Weight en numérique
dataset['Weight'] = dataset["Weight"].str.slice(0,-2).astype(float)
dataset.drop(['laptop_ID'],axis=1,inplace=True)

print(pd.concat([dataset.head(), dataset.tail()]))
print(dataset.info(),"\n\n")

#-->valeurs nulles
val_null = dataset.isnull().sum()
print("Valeurs nulles/colonne:\n",val_null)

#-->Valeurs uniques
print("\nValeurs uniques/colonne:")
for uniq in dataset.columns:
    print(uniq,":","==>",dataset[uniq].nunique(),"\n")

#-->Collection des colonnes catégoriques - numériques
print(dataset.dtypes)
numericals_col = [num for num in dataset.columns if dataset[num].dtypes =='float64']
categoricals_col = [cat for cat in dataset.columns if dataset[cat].dtypes == 'object']
dataset_num = pd.DataFrame(dataset, columns=numericals_col)
print(dataset_num)
print(numericals_col)
print(categoricals_col)

#-->Encodage
encoder  =OneHotEncoder()
col_encoded = encoder.fit_transform(categoricals_col)
datafram_encoded = pd.DataFrame(data=col_encoded, columns=encoder.get_feature_names_out(categoricals_col))
print(datafram_encoded)
 
 

