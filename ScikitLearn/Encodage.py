import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# Load dataset
data = pd.DataFrame({
    "A": ["grand", "moyen", "grand", "petit","moyen"],
    "B": [2, 4, 6, 8, 10],
    "C": [3, 6, 9, 12, 15]
})
print(data.info())

index = pd.RangeIndex(start=0 ,stop=len(data), step=1)
data.index = index

print(data)
print(data['A'].unique())
print(data['A'].value_counts())

#Encoding with Pandas
column_encoded = pd.get_dummies(data, columns=["A"])
column_encoded = column_encoded.astype(int)
print(column_encoded)

#Encoding with Scikit-learn

data2 = {'Employee id': [10, 20, 15, 25, 30],
        'Gender': ['M', 'F', 'F', 'M', 'F'],
        'Remarks': ['Good', 'Nice', 'Good', 'Great', 'Nice'],
        }
data2 = pd.DataFrame(data2)
print(data2)

#Extract categoricals columns
categoricals_columns = data2.select_dtypes(include=['object']).columns.tolist()
print(categoricals_columns) 

#Initialize OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)

#Encoding data with encoder
column_encoded = encoder.fit_transform(data2[categoricals_columns])

#Convert to DataFrame and add column names
one_hote_dtfrm = pd.DataFrame(column_encoded, columns=encoder.get_feature_names_out(categoricals_columns))

#Combine encoded and non-encoded dataframes
data_encoded = pd.concat([data2,one_hote_dtfrm], axis=1)

#Drop original categorical columns from encoded dataframe
data_encoded = data_encoded.drop(categoricals_columns, axis=1)
print(data_encoded)
