import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import warnings
import xgboost as xgb
from scipy import stats
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

#Load dataset
warnings.filterwarnings("ignore")
dataset = pd.read_csv('/Users/apple/Desktop/ML_Algorithms/Dataset/breast-cancer.csv')

#Exploring data
print(dataset.head(),"\n\n")
print(dataset.info(),'\n')
print(dataset.shape)
print("\n",dataset.columns,"\n\n")
print(dataset.describe().T,"\n\n")

#Data preprocessing
uniq_val= dataset.nunique().T
print("Uniq values by column:\n",uniq_val,"\n")
dataset.drop(columns='id',inplace=True)
#Separating columns for sorting
target_column = dataset.columns[0] #Separate target columns from features columns
features_col = dataset.columns[1:] #Selecting columns to sort
dataset_sorted = sorted(features_col, key=lambda x: x.split('_')[0])
dataset = dataset[dataset_sorted + [target_column]] #Re-unite columns
print("Dataset columns after sort:\n",dataset.columns,"\n")
def check_null_values(df):
    # Compter les valeurs nulles pour chaque colonne
    null_counts = df.isnull().sum()
    # Créer un DataFrame pour afficher les résultats 
    dtfrm_null = pd.DataFrame({
        'Column': null_counts.index,
        'Null Values': null_counts.values,
        'Percentage': (null_counts.values / len(df)) * 100  # Calculer le pourcentage de valeurs nulles
    })
    # Filtrer pour afficher seulement les colonnes ayant des valeurs nulles
    dtfrm_null = dtfrm_null[dtfrm_null['Null Values'] > 0]
    return dtfrm_null
uniq_val_target = dataset['diagnosis'].value_counts(normalize=True)
sns.barplot(x=uniq_val_target.index, y=uniq_val_target.values,hue=uniq_val_target.index)
plt.title('Distribution of class ')
plt.show()
print("Percentage of class:",uniq_val_target.round(2))

#Exploratory Data Analysis 

#Boxplot
fig ,ax = plt.subplots(6 , 5 , figsize = (15,30 ))
ax = np.array(ax).flatten()
for i, col in enumerate (dataset.columns[:-1]):
    sns.boxplot(data=dataset, x ='diagnosis', y=col, ax= ax[i],hue='diagnosis')
    ax[i].set_xlabel('Diagnosis', fontsize=10, fontweight='bold')   
    ax[i].set_ylabel(col, fontsize=10, fontweight='bold')
plt.tight_layout(w_pad=5, h_pad=5)
plt.show()

#KDE
fig ,ax = plt.subplots(6,5 , figsize = (20,40))
ax = np.array(ax).flatten()
for i, col in enumerate (dataset.columns[:-1]):
    sns.kdeplot(data=dataset,x=col, ax= ax[i],hue='diagnosis',fill=True)
    ax[i].set_xlabel(col, fontsize=10, fontweight='bold')   
    ax[i].set_ylabel(" ")
plt.tight_layout(w_pad=5, h_pad=5)
plt.show()

#Correlation
corr_matrix = dataset[features_col].corr()
plt.figure(figsize=(12, 8))  # Définir la taille du graphique
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.1f',)
plt.title('Matrice de Corrélation', fontsize=15)
plt.show()
#Remove outliers
Q1 = dataset[features_col].quantile(0.25)
Q3 = dataset[features_col].quantile(0.75)
IQR = Q3 - Q1
dataset = dataset[~((dataset[features_col] < (Q1 - 1.5 * IQR)) |(dataset[features_col] > (Q3 + 1.5 * IQR))).any(axis=1)]
print(dataset.info(),"\n\n")
encoder = LabelEncoder()
dataset['diagnosis']= encoder.fit_transform(dataset['diagnosis'])
print(dataset['diagnosis'].unique(),"\n\n") 

#Splitting in features - target
features = dataset.drop(columns='diagnosis',axis=1)
target= dataset['diagnosis']
#Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#Models 

    #Random Forest
rfc= RandomForestClassifier()
param_grid = ({
    'max_depth':list(range(1,5)),
    'min_samples_split': [1,2,3],
    'n_estimators':list(range(4,10)),
    'criterion':['gini','entropy'],
    'max_leaf_nodes':list(range(2,5))
})
grid_search = GridSearchCV(rfc,param_grid=param_grid, cv=8, n_jobs=-1)
grid_search.fit(x_train, y_train)
accuracy = round(grid_search.best_score_ *100,1)
print('Random Forest Accuracy:', accuracy,"\n")
print("Best parameters found on development set:\n",grid_search.best_params_)
#Random forest final model
best_param = grid_search.best_params_
rfc_final = RandomForestClassifier(**best_param)
rfc_final.fit(x_train, y_train)
data_predicted = rfc_final.predict(x_test)

#Rating & Visualization 
rapport = classification_report(y_test,data_predicted)
print("Classification report:\n",rapport,"\n\n")
matrix = confusion_matrix(y_test, data_predicted)
sns.heatmap(matrix, annot=True, fmt='d')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
  #Trees visualization
for tree , model in enumerate(rfc_final.estimators_):
    plt.figure(figsize=(10, 5))
    plot_tree(model, feature_names=features.columns, filled=True)
    plt.title(f"Decision Tree {tree+1}")
    plt.show()
#Features importances
features_imprt = pd.DataFrame({
    'Feature_Name':features.columns,
    'Importance':rfc_final.feature_importances_
})
dataset_sorted = features_imprt.sort_values(by='Importance',ascending=False).head(10) #Sorting feature importance and select only 10 first features
dataset_sorted['Importance']=round(dataset_sorted['Importance'] * 100,1)
index_generate = pd.RangeIndex(start=0, stop=len(dataset_sorted), step=1) 
dataset_sorted.index= index_generate
print(dataset_sorted,"\n\n")
#Visualisation of top 10 most important features
plt.figure(figsize=(10,8))
sns.barplot(data=dataset_sorted, x='Importance', y="Feature_Name",hue=index_generate,palette='rocket')
plt.title("Top 10 Most Importantant Features - Random Forest Classifier")
plt.xlabel('Importance',fontsize=15)
plt.ylabel('Feature_Name',fontsize=15)
plt.show()

    #Gradient Boosting
gbc =GradientBoostingClassifier()
param_grid = ({
        'max_depth':list(range(1,5)),
        'n_estimators':list(range(4,10)),
        'max_leaf_nodes':list(range(2,5)),
        'learning_rate':[0.01, 0.05, 0.1, 0.2]
    })
grid_search= GridSearchCV(estimator=gbc,param_grid=param_grid, cv=8,n_jobs=-1)
grid_search.fit(x_train, y_train)
accuracy = round(grid_search.best_score_ *100, 1)
print('Gradient Boosting accuracy:',accuracy,"\n")
print("Best parameters found on development set:\n",grid_search.best_params_)
#Gradient Boosting final model
final_param = grid_search.best_params_
gbc_final_model = GradientBoostingClassifier(**final_param, loss='log_loss')
gbc_final_model.fit(x_train, y_train)
prediction = gbc_final_model.predict(x_test)
    
    #Rating & Visualization
metrics = classification_report(y_test,prediction)
print("Classification report:\n",metrics,"\n\n")
matrix = confusion_matrix(y_test, prediction)
sns.heatmap(matrix, annot=True, fmt='d')
plt.title('Gradient Boosting Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

   #XGBoost
xgb_model = xgb.XGBClassifier()
param_grid = ({
    'learning_rate': [0.01,0.1,1],
    'gamma': [0,0.1,1],
   'max_depth': [3,4,5],
    'n_estimators': [10,20,30]
})
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=8, n_jobs=-1)
grid_search.fit(x_train, y_train)
accuracy = round(grid_search.best_score_ *100, 1)
print('XGBoost accuracy:',accuracy,"\n")
print("Best parameters found on development set:\n",grid_search.best_params_)
#XGBoost final model
final_param = grid_search.best_params_
xgb_final_model = xgb.XGBClassifier(**final_param)
xgb_final_model.fit(x_train, y_train)
prediction = xgb_final_model.predict(x_test)
    
    #Rating & Visualization
metrics = classification_report(y_test,prediction)
print("Classification report:\n",metrics,"\n\n")
matrix = confusion_matrix(y_test, prediction)
sns.heatmap(matrix, annot=True, fmt='d')
plt.title('XGBoost Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#Models comparison
models = [rfc_final, gbc_final_model, xgb_final_model]
model_names = ['Random Forest', 'Gradient Boosting', 'XGBoost']
results = []
for model, name in zip(models, model_names):
    scores = cross_val_score(model, features, target, cv=5, scoring='accuracy', n_jobs=-1).round(1)
    results.append({'Model': name, 'Accuracy Mean': scores.mean(), 'Accuracy Std': scores.std()})
results_df = pd.DataFrame(results)
print(results_df.round(1))
sns.barplot(x='Model', y='Accuracy Mean', data=results_df, palette='rocket')
plt.title('Models Comparison')
plt.ylabel('Accuracy (Mean)')
plt.xlabel('Models')
plt.ylim(0.8, 1)
plt.show()

#Visualization 
plt.figure(figsize=(8, 5))
sns.barplot(x='Model', y='Accuracy Std', data=results_df, palette='magma')
plt.title('Standard deviation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Models')
plt.show()








