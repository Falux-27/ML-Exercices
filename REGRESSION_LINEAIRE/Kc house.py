import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression , Lasso
from sklearn.metrics import mean_absolute_error,r2_score, mean_squared_error

#load data
dataset = pd.read_csv("/Users/apple/Desktop/ML_ALgorithms/REGRESSION_LINEAIRE/kc_house_data.csv")

#Data cleaning
dataset.drop(["id"],axis=1,inplace=True)
valeur_nulles = dataset.isnull().sum()
print(valeur_nulles)
print(dataset.info())

dataset['date']= pd.to_datetime(dataset["date"])
dataset['Year'] = dataset.date.dt.year
dataset['month'] = dataset.date.dt.month
dataset['day'] = dataset.date.dt.day
#Visualisation
sns.set_style('darkgrid')
sns.lineplot(data=dataset, y='price', x='date')
plt.title("Prix des maison en fonction des annnées")
plt.show()
sns.set_style('darkgrid')
sns.scatterplot (data=dataset, y='price', x='date')
plt.title("Prix des maison en fonction des annnées")
plt.show()

sns.set_style('whitegrid')
sns.barplot(data= dataset, x = 'bedrooms' , y= 'price', hue='bedrooms', errorbar=None)
plt.title("Prix des maison par rapport au nombre de lit")
plt.show()

sns.set_style('darkgrid')
sns.lineplot(data=dataset, x = 'yr_built', y='price')
plt.xticks(rotation = 90)
plt.title("prix des maison en fonction de l'annnée de construction")
plt.show()

ax = plt.axes(projection = '3d')
ax.scatter(dataset['bedrooms'],
           dataset['sqft_living'], 
           dataset['price'])
plt.title('Visualisation 3D des données')
ax.set_xlabel('Nombre de chambres')
ax.set_ylabel('Surface habitable')
ax.set_zlabel('Prix')
plt.show()


dataset.drop(columns=['date','Year','month','day'], axis=1, inplace=True)
print(pd.concat([dataset.head(), dataset.tail()]))

features = dataset.drop(columns=['price'])
target =dataset['price']
print(features)

scaler= StandardScaler()
features = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train,y_train)
print("L'intercept du modele:", model.intercept_)
print("Les coefs du modele:",model.coef_)

values_predict= model.predict(X_test)
print(f"Les valeurs predites  :\n",values_predict)
valeur_reell_predicts =pd.DataFrame({
    'Prix_réelles':y_test,
    'Prix_prédites':values_predict
})
print("tableau des valeur reelles - predites:\n",valeur_reell_predicts.head(20))

#Evaluation du modele

print("<------Performance modele linéaire----->")
r_squared =r2_score(y_test, values_predict)
print("R-squared:", r_squared)

MAE = mean_absolute_error(y_test, values_predict)
print("Mean Absolute Error:",MAE)

MSE= mean_squared_error(y_test, values_predict)
print("Mean Squared Error:",MSE, "\n\n")

print("<------Performance modele Lasso----->")
model_lasso = Lasso(alpha=5.0,max_iter=10000)
model_lasso.fit(X_train,y_train)
price_predict = model_lasso.predict(X_test)
print("les valeurs predites avec lasso :\n",price_predict)

r_squared_lasso = r2_score(y_test, price_predict)
print("R-squared Lasso:", r_squared_lasso)
MAE_lasso = mean_absolute_error(y_test, price_predict)
print("Mean Absolute Error Lasso:", MAE_lasso)


#Visualisation du modele
plt.scatter(y_test, values_predict)
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()], linestyle="--", c = 'pink')
plt.xlabel("Valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.title("Comparaison des valeurs réelles et prédites")
plt.show()