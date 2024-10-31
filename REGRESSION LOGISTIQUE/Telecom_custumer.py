import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score , KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score ,confusion_matrix,ConfusionMatrixDisplay, precision_score, recall_score, f1_score

#Load data
dataset = pd.read_csv("Churn.csv")
print(pd.concat([dataset.head(), dataset.tail()]))
print(dataset.dtypes)

#explore and cleaning  data
print(dataset.info())
dataset.drop(["customerID","OnlineBackup","DeviceProtection","TechSupport",
              "StreamingTV","StreamingMovies"],axis=1, inplace=True)

#Encodage
encoder = LabelEncoder()
column_to_encode = ['gender','SeniorCitizen','Partner',"Dependents","PhoneService",
        "MultipleLines","InternetService","OnlineSecurity","Contract",
        "PaperlessBilling","PaymentMethod", "Churn"]

for col in column_to_encode:
    dataset[col] = encoder.fit_transform(dataset[col])
print(pd.concat([dataset.head(), dataset.tail()]))

#Mise en echelle  
dataset['TotalCharges'] = dataset['TotalCharges'].replace(' ', np.nan)
dataset['TotalCharges'] = dataset['TotalCharges'].astype(float)
dataset.fillna(value=dataset["TotalCharges"].mean(),inplace=True)
scaler = StandardScaler()
dataset[['tenure','MonthlyCharges','TotalCharges']] = scaler.fit_transform(dataset[['tenure','MonthlyCharges','TotalCharges']])
print(pd.concat([dataset.head(), dataset.tail()]))

#Separation du dataset en features et target  
features = dataset.drop(columns=["Churn"])
target = dataset["Churn"]

#Cross-Validation
classifier = LogisticRegression()
fold_numb = 10
k_folds = KFold(n_splits=fold_numb, shuffle=True,random_state=42)
cross_val_result = cross_val_score(classifier,features,target,cv=fold_numb)
print(f'Cross-Validation Results (Accuracy): {cross_val_result}')
print(f'Cross_validation moyenne: {cross_val_result.mean()}')

#Separation des donnees en Train et Test:
x_train, x_test,y_train,y_test = train_test_split(features, target, test_size=0.2, random_state=42)

print("Les donnes de test:",x_test,"\n\n",y_test,"\n\n")

#Entrainement du modele
model_classifier = classifier.fit(x_train,y_train)
print("L’ordonnée à l'origine:", model_classifier.intercept_)
print("Les coefficents du modele:", model_classifier.coef_)

#Prediction sur les donnees de test
y_predict = model_classifier.predict(x_test)
print("test:\n",y_predict)

#Comparaison des valeurs reelles-predites
tab_comparative = pd.DataFrame({
    "Valeurs reelles":y_test,
    "Valeurs predites":y_predict
})
print("tableau des valeur reelles - predites:\n",tab_comparative.head(20),"\n\n")


#Evaluation de la performance du modele
#Confusion-Matrix
matrix =confusion_matrix(y_test, y_predict)
print("La matrice de confusion:\n",matrix,"\n")
#graphe
fig  =ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[0,1])
fig.plot()
plt.show()

#Accuracy_score:
precision = accuracy_score(y_test,y_predict)

#Recall
rappel =recall_score(y_test,y_predict)

#F1-score
f1_score = f1_score(y_test, y_predict)

print("Accurracy-score = ", precision)
print("Recall-score = ", rappel)
print("F1-score = ", f1_score)


