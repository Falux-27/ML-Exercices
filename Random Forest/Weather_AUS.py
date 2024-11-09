import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.model_selection import GridSearchCV 
from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.metrics import confusion_matrix

# Load dataset
dataset = pd.read_csv('/Users/apple/Desktop/ML_Algorithms/REGRESSION LOGISTIQUE/weatherAUS.csv')
print(dataset.head(),"\n\n")

#Exploring data
print(dataset.info(),'\n')
print(dataset.shape)

#Data preprocessing
dataset.drop(columns='Date',inplace=True)
print(dataset.isnull().sum(),'\n')
val_uniq_target = dataset['RainTomorrow'].unique()
print("Les valeurs uniques pour la colonne RainTomorrow:",val_uniq_target,"\n")
proportion_target = dataset['RainTomorrow'].value_counts(normalize=True).plot.bar()
plt.title('Distribution des jours avec et sans pluie')
plt.show()

percentage_null = []
threshold = 0.4
for x in dataset.columns:
    percent = dataset[x].isnull().mean()
    print(f"{x}: {percent:.2f}")
    if percent >= threshold:
        percentage_null.append(x)
print(f"Colonnes avec plus de 40% des données manquantes:{percentage_null}\n")
 
cat_col = [col for col in dataset.columns if dataset[col].dtypes == 'object']
print(f'Colonnes  catégorielles :\n')
for col in cat_col:
    print(f"{col}:  {dataset[col].dtypes}","\n")
num_col = [col for col in dataset.columns if col not in cat_col]
print(f'Colonnes numériques :\n')
for z in num_col:
    print(f"{z} :  {dataset[z].dtypes}","\n")
for x in cat_col:
    print(f"{x}: {dataset[x].unique().tolist()}")

#Correlation
corr = dataset[num_col].corr()
plt.figure(figsize=(15, 15))
sns.heatmap(corr, annot=True, square=True)
plt.show()

#Matrice de correlation
matrix_corr = corr.abs()
print(matrix_corr,"\n\n")
#Matrice triangulaire
mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
tri_matrix = corr.where(mask)
print(tri_matrix,"\n\n")
#Seuil
treshold = 0.85
high_corr_var = [col for col in tri_matrix.columns if any(tri_matrix[col] >= treshold)]
print(f"Colonnes avec une corrélation supérieure à {treshold}:\n {high_corr_var}\n")
#Paire de variable fortement corrélées
pairs = [(col,row)for col in tri_matrix.columns for row in tri_matrix.index if tri_matrix[col][row]>= treshold]
print(f"Paires de variables fortement corrélées:\n {pairs}\n")
#Supprimer certains colonnes 
dataset.drop (columns=['Pressure9am','Temp9am','Temp3pm'],inplace=True)
print(dataset.columns,"\n\n")

#Handling missing values
simple_imput = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
numerical_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
for col in dataset.columns:
    if dataset[col].dtype == 'object':
        dataset[col] = simple_imput.fit_transform(dataset[[col]]).ravel()
col_num_to_imput = [col for col in dataset.columns if dataset[col].dtype in ['float64','int64']]
for col in col_num_to_imput:
    dataset[col] = numerical_imputer.fit_transform(dataset[[col]]).ravel()
print(dataset.isnull().sum())

#Data Splitting
features= dataset.drop(columns='RainTomorrow')
target = dataset['RainTomorrow']
#Encoding categoricals
for col in features.columns:
    if features[col].dtype== 'object':
        encoder = LabelEncoder()
        features[col] = encoder.fit_transform(features[col])
target = target.map({'No':0,'Yes':1})

x_train, x_test, y_train, y_test = train_test_split(features,target, test_size=0.2, random_state=42)
#Oversampling
smote =SMOTE()
xtrain_sm , ytrain_sm = smote.fit_resample(x_train,y_train)

                            #Training models
param_grid = {
    'n_estimators': [50, 100],
   'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'gamma': [0.0, 0.1, 0.2],
    'lambda':[0.0,0.1,0.2]  ,
    "objective":["binary:logistic"]
}
model = xgb.XGBClassifier()
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=1)
grid_search.fit(xtrain_sm, ytrain_sm)
print(f'La meilleur score:{grid_search.best_score_}',"\n")
print(f'Les meilleurs paramètres :{grid_search.best_params_}')

#XGBOOST
best_params = grid_search.best_params_
xg_model = xgb.XGBClassifier(**best_params)
xg_model.fit(xtrain_sm, ytrain_sm)
y_pred = xg_model.predict(x_test)

#Evaluation
f1_score_val = f1_score(y_test, y_pred, average='weighted')
print(f'Le F1-score est de :{f1_score_val}')
confusion_matrix_val = confusion_matrix(y_test, y_pred)
sns.heatmap(confusion_matrix_val,annot=True,fmt='d')
plt.show()


 



 




