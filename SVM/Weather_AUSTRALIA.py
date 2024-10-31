import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split ,cross_val_score , StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC

#Load dataset
dataset =pd.read_csv("/Users/apple/Desktop/ML_Algorithms/SVM/weatherAUS.csv")
print(dataset.head(),"\n")
#Exploring data
print(dataset.info(),"\n")
#Number of NaN values for each columns
print(dataset.isnull().sum(),"\n")

#Columns with missing values
val_null =[col for col in dataset.columns if dataset[col].isnull().sum()>0]
print(val_null,"\n")

#Handling missing values
missing_val = []
for col in val_null:
    val_null_col = dataset[col].isnull().sum()
    missing_val.append(val_null_col)
datafrm_null = pd.DataFrame({
    'Column': val_null,
    'Values': missing_val
})
print(datafrm_null,"\n")

#Visualisation 
for col in datafrm_null:
    sns.barplot(datafrm_null, x= 'Column', y='Values', hue='Column')
    plt.xticks(rotation = 45)
    plt.title(f"Distribution of missing values")
#plt.show()

#Feature engineering

dataset.drop(columns=['Date'],axis=1, inplace=True)
    #Filling with missing values
for col in dataset.columns:
    if dataset[col].dtype in ['int64','float64']:
        dataset[col].fillna(dataset[col].mean(),inplace=True)
    elif dataset[col].dtype == 'object':
        dataset[col].fillna(dataset[col].mode()[0], inplace=True)
        
#Splitting dataset into features -target
features = dataset.drop(columns=['RainTomorrow'])
target = dataset['RainTomorrow']
print(features)

#Exploring target column
val_uniq =dataset['RainTomorrow'].value_counts()
print("Nombre d'instance/classe:\n", val_uniq)
percentage_classe = dataset['RainTomorrow'].value_counts(normalize=True)
print("Pourcentage des classes:",percentage_classe.round(2))
target_dtfrm = pd.DataFrame({
    'classe':val_uniq.index,
    'valeur':val_uniq.values
})
sns.barplot(data=target_dtfrm, x= 'classe', y='valeur', hue='classe')
plt.title('Distribution des instances par classe')
plt.xlabel('Classe')
plt.ylabel('Nombre d\'instances')
#plt.show()

#Encoding categoricals columns
    #--->Encoding features
encoder =OneHotEncoder(sparse_output=False,drop='first')
categorical_col =[col for col in features if features[col].dtype not in ['float64', 'int64']]
print("Colonnes cat:",categorical_col,"\n")
encoded_col = encoder.fit_transform(features[categorical_col]).astype(int)
data_encoded = pd.DataFrame(encoded_col, columns=encoder.get_feature_names_out(categorical_col))
features_encoded = pd.concat([features.drop(columns=categorical_col, axis=1), data_encoded], axis=1)

    #---->Encoding Target
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)
print(target_encoded)

    #---->Scaling numericals columns
numericals_col = [col for col in features_encoded if features_encoded[col].dtype
                  in ['float64', 'int64']and col not in 
                  categorical_col and col not in data_encoded]
print("Colonnes num:",numericals_col,"\n")
scaler = MinMaxScaler(feature_range=(1,2))
features_encoded[numericals_col]=scaler.fit_transform(features_encoded[numericals_col]).round(2)
print(features_encoded.columns)

#Splitting dataset into train and test set
x_train, x_test, y_train, y_test = train_test_split(features_encoded, target_encoded, test_size=0.2, random_state=42)

#Oversampling
smote =SMOTE(random_state=1)
x_train_sm, y_train_sm = smote.fit_resample(x_train, y_train)

#Logistic Regression model 
Logistic_model = LogisticRegression(max_iter=10000,penalty='l1', solver='liblinear')
Logistic_model.fit(x_train_sm, y_train_sm)
origine = Logistic_model.intercept_
coef = Logistic_model.coef_
class_predicted = Logistic_model.predict(x_test)
matrix =confusion_matrix(y_test, class_predicted)
sns.heatmap(matrix, annot=True, fmt='g')#
plt.title("Matrice de confusion - Logistic Regression")
plt.xlabel("Prédiction")
plt.ylabel("Valeur réelle")
plt.show()
print('L\'equation du modele:\n',origine,"\n",coef,"\n")
rapport = classification_report(y_test, class_predicted)

 



    