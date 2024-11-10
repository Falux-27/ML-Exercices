import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import GridSearchCV ,StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer,SimpleImputer


# Load dataset
dataset = pd.read_csv('/Users/apple/Desktop/ML_Algorithms/REGRESSION LOGISTIQUE/bank-additional-full.csv',delimiter=';')

#Exploring data
print(dataset.head(),"\n\n")
print(dataset.info(),'\n')
print(dataset.shape)

#Data preprocessing
column_ = dataset.columns
print(column_,"\n\n")

liste_columns = dataset.columns
def valeur_uniq (x):
    result = []
    for col in dataset.columns:
        if dataset[col].dtype == 'object':
            uniq_val = dataset[col].unique()
            result.append(f"{col}:{uniq_val}")
        else:
            uniq_val = "Pas applicable"
    return "\n\n".join(result)

print(valeur_uniq(liste_columns) )
cat_var = [cat for cat in dataset.columns if dataset[cat].dtype == 'object']
def nbr_valeur_uniq (x):
    liste_ = []
    for col in cat_var:
        nbr_uniq = dataset[col].value_counts()
        liste_.append(f"{col}: {nbr_uniq}")
    else:
        nbr_uniq = 'Pas applicable'
    return "\n\n".join(liste_)

print(nbr_valeur_uniq(cat_var))
def visualization (x):
    for col in dataset.columns:
        if dataset[col].dtype == 'object':
            count_val = dataset[col].value_counts()
            sns.barplot( x= count_val.index,y=count_val.values,palette='Set1')
            plt.title(f"Distribution des valeurs uniques de :{col}")
            plt.xlabel(col)
            plt.ylabel("Nombre de valeurs")
            plt.xticks(rotation=45)
            plt.show()
            
z = 'unknown'
for col in dataset.columns:
    if dataset[col].dtype == 'object':  
        if z in dataset[col].values:  # Vérifiez si 'unknown' est présent
            mode_value = dataset[col].mode()[0]  # Trouvez la mode de la colonne
            dataset[col].replace(z, mode_value, inplace=True)  # Remplacez 'unknown' par la mode
 
dataset['education']= dataset['education'].replace({
    'university.degree':'university_degree',
    'high.school':'high_school',
    'basic.9y':'basic',
    'basic.6y':'basic',
    'basic.4y':'basic',
    'professional.course':'Boot_camp'
})
dataset.drop(columns=['month','day_of_week'],inplace=True)

numerical_col =  [col for col in dataset.columns if dataset[col].dtype in ['int64','float64']]
matrix= dataset[numerical_col].corr()
sns.heatmap(matrix, annot=True, cmap='coolwarm')
#plt.show()

triangle_matrix = np.triu(np.ones(matrix.shape),k=1).astype(bool)
print(triangle_matrix,"\n\n")
tri_matrix = matrix.where(triangle_matrix)
print(tri_matrix,"\n\n")
treshold = 0.9
hig_corr_vars = [(col,row) for col in tri_matrix.columns for row in tri_matrix.index if tri_matrix[col][row]>= treshold]
print(f'Variables fortement corrélées:{hig_corr_vars}\n')
print(dataset['y'].value_counts(),"\n")
dataset.drop(columns='euribor3m',inplace=True)

features = dataset.iloc[:,:-1]
target = dataset['y']

encoder = OneHotEncoder(sparse_output=False)
col_to_encod = [col for col in features.columns if features[col].dtype == 'object']
print('Colonne categorielle:',col_to_encod,"\n\n")
features_encoded = encoder.fit_transform(features[col_to_encod])
dtfrm_encoded = pd.DataFrame(features_encoded,columns=encoder.get_feature_names_out(col_to_encod))
features = features.drop(columns=col_to_encod,axis=1)
features = pd.concat([features,dtfrm_encoded],axis=1)
target = target.map({"no":0,"yes":1})
print(features.columns)

#Splitting dataset
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
smote = SMOTE()
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

            #Models
#Decision Tree
    #Grid_Search
params = ({
    'criterion':['gini','entropy'],
    'max_depth':[2,4,6,8],
    'min_samples_split':[1,2,3],
    'min_samples_leaf':[1,2],
    'max_leaf_nodes':[3,5,8],
    'ccp_alpha': [0.0, 0.01, 0.1]
})
model = DecisionTreeClassifier()
grid = GridSearchCV(estimator=model , param_grid=params,cv=StratifiedKFold(n_splits=5),n_jobs=-1)
grid.fit(x_train_smote, y_train_smote)
print("Meilleur score des plis:",grid.best_score_,"\n")
print("Meilleurs paramètres:",grid.best_params_,"\n\n")

#Modele finale
best_param = grid.best_params_
final_model = DecisionTreeClassifier(criterion=best_param['criterion']
                                     ,max_depth=best_param['max_depth']
                                     ,min_samples_split=best_param['min_samples_split']
                                     ,min_samples_leaf=best_param['min_samples_leaf']
                                     ,max_leaf_nodes=best_param['max_leaf_nodes']
                                     ,ccp_alpha=best_param['ccp_alpha']
                                     )
final_model.fit(x_train_smote, y_train_smote)
values_predicted = final_model.predict(x_test)

#Evaluation
rapport = classification_report(y_test,values_predicted)
print("Classification report:\n",rapport)
score = accuracy_score(y_test,values_predicted)
print("Le score du modele :",score,"\n")

#Confusion matrix
matrix = confusion_matrix(y_test,values_predicted)
sns.heatmap(matrix,annot=True,fmt='d')
plt.title('Decision Tree Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

#Plot tree
plot_tree(final_model, feature_names=features.columns, filled=True)
plt.show()

