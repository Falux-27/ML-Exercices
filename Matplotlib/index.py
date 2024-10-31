import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

dtfrm = pd.DataFrame({
    "Duration": [10, 20,30, 40],
  "Calories": [40.1, 50.5,65.1,74.8]
})
plt.plot(dtfrm["Duration"],dtfrm["Calories"])
plt.show()

#Tracer sans ligne:
tab1 = pd.DataFrame({
    "a":[32,12,5,62],
    "z":[26,43,60,55]
})
plt.plot(tab1["a"],tab1["z"],"o")
plt.show()

#Marqueur:
data = pd.DataFrame({
    "points": [120,345,198,30],
    "total":[654,148,498,348]
})
plt.plot(data["points"],data["total"],marker = '*')
plt.show()

#Style de ligne:
x= np.array([10,20,30,40])
y= np.array([50,30,45,25])
plt.plot(x,y,linestyle = 'dotted')
plt.show()

#Largeur de ligne:
y= np.array([50,30,45,25])
plt.plot(y , linewidth = '12')
plt.show()


#Lignes multiples:
Names = pd.DataFrame({
  "firstname": ["Sally", "Mary", "John"],
  "age": [50, 40, 30]
})
plt.plot(Names["firstname"])
plt.plot(Names["age"])
plt.show()