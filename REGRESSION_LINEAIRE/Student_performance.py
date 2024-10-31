import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns
from sklearn.calibration import LabelEncoder
from sklearn.metrics import r2_score , mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
 

#Chargement des donnees
data = pd.read_csv("/Users/apple/Desktop/ML_Algorithms/REGRESSION_LINEAIRE/Student_Performance.csv")
print("Donnees sans modification:\n\n",data.head(),"\n\n")

#Encodage de label des donnees catégorielle
label_encoder = LabelEncoder()  #Créer l'objet d'encodage
data['Extracurricular Activities'] = label_encoder.fit_transform(data['Extracurricular Activities'])
print(data.head(),"\n\n")

#Visualisation des donnees 
#sns.pairplot(data)
#plt.show()

#Separation du dataset en features et target
feature = data.drop(columns=["Performance Index"])
target = data["Performance Index"]

#Separation des donnees en train et test:    
xtrain, xtest, ytrain, ytest = train_test_split(feature, target,test_size=0.2, random_state=42)
print("Les donnees de test:\n", xtest)
print(" ")

#Entraînement du model
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

#L'equation:
print("L’ordonnée est :",regressor.intercept_,"\n\n")
print("Les coefficients sont : ",regressor.coef_,"\n")

#Prediction  sur les donnes de test
y_predict = regressor.predict(xtest)
print("Les performance predites:\n", y_predict)

#Evaluation du modele
R_carre = r2_score(ytest,y_predict)
MsE =mean_squared_error(ytest, y_predict) 

print("R-squared --->", R_carre)
print("Mean Squared Error --->", MsE)

#Représentation graphique:
plt.scatter(ytest, y_predict)
plt.plot([ytest.min(), ytest.max()],[ytest.min(), ytest.max()], c = "g", linestyle="--")
plt.xlabel("performance réelles")
plt.ylabel("performance prédites")
plt.title("Comparaison des performances réelles et prédites")
plt.show()
