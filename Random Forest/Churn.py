import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,mean_squared_error,r2_score,confusion_matrix,classification_report
from sklearn.ensemble import  RandomForestClassifier,VotingClassifier 
from sklearn.model_selection import GridSearchCV 
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer,SimpleImputer
 

# Load dataset
dataset = pd.read_csv('/Users/apple/Desktop/ML_Algorithms/REGRESSION LOGISTIQUE/Churn.csv')
print(dataset.head(),"\n\n")
#Exploring data
print(dataset.info(),'\n')
print(dataset.shape)
 
 #Data preprocessing
percent_target = dataset['Churn'].value_counts(normalize=True)
print("Pourcentage des classes:", percent_target.round(2))
dataset['Churn'].value_counts().plot.bar(color=['grey' , 'pink'])
#plt.show()
dataset['Churn']=dataset['Churn'].map({'No':0,'Yes':1})
print(dataset['Churn'].unique(),"\n")

for x in dataset['TotalCharges']:
    if x==' ':
        dataset['TotalCharges'].replace(' ', np.nan, inplace=True)
imputer = KNNImputer(n_neighbors=3)
dataset['TotalCharges'] = imputer.fit_transform(dataset[['TotalCharges']]).ravel()
dataset['TotalCharges'] = dataset['TotalCharges'].astype(float)

dataset.drop(columns='customerID',inplace=True)
dataset.drop(columns='PaymentMethod',inplace=True)
liste_to_drop=['StreamingMovies','MultipleLines','OnlineSecurity',
                      'DeviceProtection','OnlineBackup','TechSupport']
for col in dataset.columns :
    if col in liste_to_drop:
        dataset.drop(columns=col,inplace=True)
print(dataset.columns,"\n")
    
cat_columns =[col for col in dataset.columns if dataset[col].dtypes == 'object']
print(f'Colonnes catégorielles : \n {cat_columns}\n')
num_columns = [col for col in dataset.columns if col not in cat_columns]
print(f'Colonnes numériques : \n {num_columns}\n')

matrix = dataset[num_columns].corr()
sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

features = dataset.iloc[:,:-1]
target = dataset.iloc[:,-1]
encoder = OneHotEncoder(sparse_output=False)
columns_to_encode = encoder.fit_transform(features[cat_columns])
dtfrm_encoded = pd.DataFrame(columns_to_encode,columns=encoder.get_feature_names_out(cat_columns))
features = features.drop(columns=cat_columns,inplace=True)
features = pd.concat([features,dtfrm_encoded],axis=1)
 
 #Splitting the dataset into the training set and test set
x_train,x_test,y_train,y_test = train_test_split(features,target,test_size=0.2,random_state=42)
#Oversampling
over_sample = SMOTE()
x_train_sm , y_train_sm =over_sample.fit_resample(x_train,y_train)

                                #Models

#Random Forest classifier
params = ({
    'n_estimators':[100,200,300],
    'max_depth':[1,2,3],
    'min_samples_split':[2,4],
    'min_samples_leaf':[1,2],
    'max_leaf_nodes':[2,3,4]
})
model = RandomForestClassifier()
grid = GridSearchCV(model,param_grid=params,cv=8,n_jobs=-1,verbose=2)
grid.fit(x_train_sm,y_train_sm)
print("Meilleur score des plis:",grid.best_score_,"\n")
print("Meilleurs paramètres:",grid.best_params_,"\n\n")
#Modele finale
best_param = grid.best_params_
final_model = RandomForestClassifier(
    n_estimators=best_param['n_estimators'],
    oob_score=True,
    n_jobs=-1,
    max_depth=best_param['max_depth'],
    min_samples_split=best_param['min_samples_split'],
    min_samples_leaf=best_param['min_samples_leaf'],
    max_leaf_nodes=best_param['max_leaf_nodes'] 
)
final_model.fit(x_train_sm,y_train_sm)
y_pred = final_model.predict(x_test)
rapport = classification_report(y_test,y_pred)
print("Classification report:\n",rapport)
matrix = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:\n",matrix)
sns.heatmap(matrix, annot=True, fmt='d')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()









