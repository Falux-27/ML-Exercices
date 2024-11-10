import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.impute import KNNImputer,SimpleImputer

# Load dataset
data  = pd.read_csv("/Users/apple/Desktop/ML_Algorithms/Dataset/clinvar_conflicting.csv")
# Exploring the data
print(data.head())
print(data.info(),"\n")
print(data.shape)
print("\n",data.columns,"\n\n")

    #Data preprocessing

#Handling missing values
val_null = data.isnull().sum()
print("Number of null values in each column:\n",val_null)
seuil = 0.5
total_rows = len(data)
list_col_drop = []
for col in data.columns:
    percentage_null = val_null [col] / total_rows
    if percentage_null > seuil:
        list_col_drop.append(col)
        data.drop(columns=[col], inplace=True)
print("List of columns dropped:\n",list_col_drop,"\n")
print("\nAfter dropping columns with more than 50% of null values:\n",data.columns,"\n")
imputer = KNNImputer(n_neighbors=3)
numericals_cols = [col for col in data.columns if data[col].dtype in ['float64','int64']]
column_to_fill = [x for x in numericals_cols if data[x].isnull().sum() > 0 ]
print("Numericals columns with NaN values:\n",column_to_fill,"\n\n")
data[column_to_fill]= imputer.fit_transform(data[column_to_fill])
print("After imputation:\n",data.isnull().sum(),"\n")
#Categoricals values
imputer_cat = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
categoricals_col = [cat_col for cat_col in data.columns if cat_col not in numericals_cols]
print("categoricals columns:",categoricals_col,"\n\n")
cat_cat_to_fill = [zx for zx in categoricals_col if data[zx].isnull().sum() > 0 ]
print("Categoricals columns with NaN values:\n",cat_cat_to_fill,"\n\n")
data [ cat_cat_to_fill]= imputer_cat.fit_transform(data[cat_cat_to_fill])
print("After all imputations:\n",data.isnull().sum(),"\n")

        #Visualization
#Distribution of numericals values
#sns.pairplot(data[numericals_cols],hue=data['CLASS'])
#plt.title("distribution of the numerical columns")
#plt.show()

#Distribution of values class
#sns.set_style('darkgrid')
#sns.countplot(data['CLASS'])
#plt.title("Total par classe")
#plt.show()

#Checking outliers
#plt.figure(figsize=(15, 8))
#sns.boxplot(data=data[numericals_cols])
#plt.xticks(rotation=90)
#plt.show()

#Correlation matrix
matrix = data[numericals_cols].corr()
#sns.heatmap(matrix,annot=True, cmap='coolwarm',fmt='.2f')
#plt.show()
#Removing Highly correlated Variables
mask = np.triu(np.ones(matrix.shape),k=1).astype(bool) #This line creates a triangular matrix of the size of 
#the correlation matrix that keeps only the values above the diagonal & fill it with boolean values
print("Mask:\n",mask,"\n")
tri_matrix = matrix.where(mask) # this line filters the correlation matrix by the mask and keeps only the values
#that are at the top of the diagonal, i.e. that are True
print("Triangular matrix:\n",tri_matrix,"\n")
treshold = 0.80
highly_corr = [(col,row) for col in matrix.columns for row in matrix.index if tri_matrix[col][row] >= treshold]
print("Highly correlated pairs:\n",highly_corr,"\n")
#Dropping highly correlated columns
data.drop (columns=['AF_TGP','AF_EXAC','CADD_RAW'],inplace=True)
print(data.columns,"\n\n")


"""  #Splitting data
features = data.drop(columns=['CLASS'],axis=1)
target = data['CLASS']
#Encoding variables 
encoder = OneHotEncoder(sparse_output=False)
data[categoricals_col] = data[categoricals_col].astype(str)
encoded_features = encoder.fit_transform(data[categoricals_col])
feature_encoded = pd.DataFrame(encoded_features,columns=encoder.get_feature_names_out(categoricals_col))
full_features = pd.concat([features,feature_encoded],axis=1)
print(full_features.head(10))
"""




 

