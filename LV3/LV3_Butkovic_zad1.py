import pandas as pd

mtcars = pd.read_csv('C://Users//bakib//Desktop//Coding//New folder//mtcars.csv')

print(mtcars.sort_values(by='mpg').head(5))

print(mtcars[(mtcars.cyl==8)].sort_values(by='mpg').tail(3))
print(mtcars[mtcars.cyl==6]["mpg"].mean())

print(mtcars[(mtcars["cyl"]==4) & (mtcars["wt"]>=2.0) & (mtcars["wt"]<=2.2)].mpg.mean())

print(mtcars["am"].value_counts())


print(len(mtcars[(mtcars["am"]==0) & (mtcars["hp"]>100)]))


print(mtcars["wt"] * 1000 * 0.453592)

