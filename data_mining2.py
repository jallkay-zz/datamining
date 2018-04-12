import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
# Import data
mydata = pd.read_csv("dataset.csv")
# Visualise data
print mydata.describe()
print ("Empty values: {}.".format(mydata.isnull().values.any()))

diabetes   = mydata[mydata.Class == "Diabetes"]
nodiabetes = mydata[mydata.Class == "DR"] 

data = [diabetes.PressureA, nodiabetes.PressureA]

plt.figure()
ax1 = plt.subplot(121)
plt.boxplot(data, labels = ["Diabetes", "No Diabetes"])


ax2 = plt.subplot(122)
pt = sns.kdeplot(diabetes.Tortuosity, shade=True, label="Diabetes")
pt = sns.kdeplot(nodiabetes.Tortuosity, shade=True, label="No Diabetes")

plt.show()
print mydata