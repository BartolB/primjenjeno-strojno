import matplotlib.pyplot as plt
import pandas as pd

mtcars = pd.read_csv('C://Users//bakib//Desktop//Coding//New folder//mtcars.csv')

mpg=mtcars.groupby("cyl")["mpg"].mean()
plt.figure()
plt.bar(mpg.index, mpg.values)
plt.title("Prosječna potrošnja po cilindrima")
plt.xlabel("Cilindri")
plt.ylabel("MPG")
plt.show()


data_4 = mtcars[mtcars["cyl"] == 4]["wt"]
data_6 = mtcars[mtcars["cyl"] == 6]["wt"]
data_8 = mtcars[mtcars["cyl"] == 8]["wt"]

plt.figure()
plt.boxplot([data_4, data_6, data_8], labels=["4", "6", "8"])
plt.title("Distribucija težine po cilindrima")
plt.xlabel("Cilindri")
plt.ylabel("Težina (1000 lbs)")
plt.show()


auto = mtcars[mtcars["am"] == 0]["mpg"]
manual = mtcars[mtcars["am"] == 1]["mpg"]

plt.figure()
plt.boxplot([auto, manual], labels=["Automatski", "Ručni"])
plt.title("Potrošnja po tipu mjenjača")
plt.ylabel("MPG")
plt.show()


plt.figure()
plt.scatter(mtcars[(mtcars["am"] == 0)]["hp"],
            mtcars[(mtcars["am"] == 0)]["qsec"],
            color="blue")
plt.scatter(mtcars[(mtcars["am"] == 1)]["hp"],
            mtcars[(mtcars["am"] == 1)]["qsec"],
            color="red")

plt.xlabel("Snaga (hp)")
plt.ylabel("Ubrzanje (qsec)")
plt.title("Snaga vs ubrzanje")

plt.show()

