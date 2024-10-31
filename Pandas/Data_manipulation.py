import pandas as pd
#AGRÉGATIONS DE DONNÉES

    #GROUP BY:
"""est utilisée pour diviser les données en groupes en fonction de certaines clés, 
puis appliquer des fonctions d'agrégation comme la somme, la moyenne ect.."""

dtfrm = pd.DataFrame(data = {
    'Key': ['A', 'B', 'A', 'B', 'A'],
    'Value': [10, 20, 30, 40, 50]
})
print(dtfrm)
#Regroupement par 'Key'
grouped = dtfrm.groupby('Key')
print("After regroupement: \n",grouped,"\n")
#Calculer la somme pour chaque groupe:
sum_grouped = grouped.sum()
print("Sum of each group: \n",sum_grouped,"\n")

#Regrouper avec plusieurs colonnes
dtfrm2 = pd.DataFrame(
    {'Class': ['A', 'B', 'A', 'B', 'A', 'B'],
        'Gender': ['Male', 'Male', 'Female', 'Female', 'Male', 'Female'],
        'Math_Score': [85, 92, 78, 89, 90, 86],
        'English_Score': [88, 94, 80, 92, 92, 88]
    })
print(dtfrm2,"\n")

# Groupement par 'Class' et 'Gender'
group = dtfrm2.groupby(['Class','Gender'])
#aggregation
stats = group['Math_Score'].mean()
print("Mean Math Score by Class and Gender: \n", stats,"\n")

stat2 = group['English_Score'].mean()
print("Mean English Score by Class and Gender: \n", stat2)

 