import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("mtcars.csv", delimiter=",", skiprows=1)

mpg = data[:,0]   # potrošnja goriva
cyl = data[:,1]   # broj cilindara
hp = data[:,3]    # konjske snage
wt = data[:,5]    # težina vozila


plt.scatter(hp, mpg)
plt.xlabel("Konjske snage")
plt.ylabel("Potrošnja")
plt.title("Potrošnja vs snaga")
plt.show()


plt.scatter(hp, mpg, s=wt*100)
plt.xlabel("Konjske snage")
plt.ylabel("Potrošnja")
plt.title("Potrošnja vs snaga (veličina = težina)")
plt.show()


# minimalna, maksimalna i srednja potrošnja
print("Minimalna potrošnja:", np.min(mpg))
print("Maksimalna potrošnja:", np.max(mpg))
print("Srednja potrošnja:", np.mean(mpg))


# filtriranje automobila sa 6 cilindara
mpg6 = mpg[cyl == 6]

print("Min potrošnja (6 cilindara):", np.min(mpg6))
print("Max potrošnja (6 cilindara):", np.max(mpg6))
print("Mean potrošnja (6 cilindara):", np.mean(mpg6))
