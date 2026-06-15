import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ucitavanje ociscenih podataka
df = pd.read_csv(r'E:\Coding_faks\Coding\New folder\cars_processed.csv')
print(df.info())

# razliciti prikazi
sns.pairplot(df, hue='fuel')

sns.relplot(data=df, x='km_driven', y='selling_price', hue='fuel')
df = df.drop(['name','mileage'], axis=1)

obj_cols = df.select_dtypes(object).columns.values.tolist()
num_cols = df.select_dtypes(np.number).columns.values.tolist()

fig = plt.figure(figsize=[15,8])
cols_count = len(obj_cols)
rows = (cols_count + 1) // 2
for col in range(cols_count):
    plt.subplot(rows, 2, col+1)
    sns.countplot(x=obj_cols[col], data=df)

sns.boxplot(data=df, x='fuel', y='selling_price')

df['selling_price'].hist(grid=False)

tabcorr = df[num_cols].corr()
sns.heatmap(df[num_cols].corr(), annot=True, linewidths=2, cmap= 'coolwarm') 

plt.show()
