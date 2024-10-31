import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from sklearn.ensemble import  RandomForestClassifier
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import mean_squared_error, r2_score
warnings.filterwarnings('ignore')

# Load dataset
dataset = pd.read_csv('/Users/apple/Desktop/ML_Algorithms/Decision Tree/heart_attack_prediction_dataset.csv')
print(dataset.head(), "\n\n")

#Exploring data
print(dataset.info(),"\n")
print(dataset.isnull().sum(),"\n")
print(dataset.describe(),"\n")

#Data preprocessing
print(dataset['Heart Attack Risk'].value_counts(normalize=True))
cat_columns = [cat for cat in dataset.columns if dataset[cat].dtype == 'object']
print(f'Les colonnes catégorielles :{cat_columns}')
for x in cat_columns:
    encoder = LabelEncoder()
    dataset[x] = encoder.fit_transform(dataset[x])
print(dataset.info(),"\n")
print(dataset.columns,"\n")
dataset.drop(columns='Patient ID')

corr_matrix = dataset.corr()
sns.heatmap(data=corr_matrix,annot=True, cmap='coolwarm', fmt='.2f' )
plt.show()

#Splitting in features - target
features = dataset.iloc[:,:-1]
target = dataset.iloc[:,-1]

#Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#Oversampling
over_sample = SMOTE()
xtrain_sm, ytrain_sm = over_sample.fit_resample(x_train,y_train)

#GridSearch
"""

param_grid= ({
    'bootstrap':[True,False],
    'n_estimators':[25,50,75,100,125,150,175,200],
    'max_depth':[1,2,3,4,5],
    'min_samples_split':[2,3,4,5],
    'min_samples_leaf':[1,2,3],
    'max_leaf_nodes':[1,2,3],
})
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(),
    param_grid=param_grid,
    cv=5,
    verbose=2,
    n_jobs=-1
)
grid_search.fit(xtrain_sm, ytrain_sm)
print(f'Meilleur-score: {grid_search.best_score_:.2f}')
print(f'Meilleurs paramètres: { grid_search.best_params_}')

"""
#Model 
rand_forest = RandomForestClassifier(n_estimators=30,random_state=0,oob_score=True,verbose=1)
rand_forest.fit(xtrain_sm, ytrain_sm)
y_pred = rand_forest.predict(x_test)

#Evaluation
f1_score_val = f1_score(y_test, y_pred, average='weighted')
print(f'F1 Score: {f1_score_val:.2f}')

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse:.2f}')

r2 = r2_score(y_test, y_pred)
print(f'R-squared (R2): {r2:.2f}')

obb_score =rand_forest.oob_score_
print(f'Out-of-Bag Score: {obb_score:.2f}')



