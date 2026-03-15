brojevi = []

while True:
    unos = input("Unesite broj: ")

    if unos == "Done":
        break

    try:
        broj = float(unos)
        brojevi.append(broj)
    except:
        print("Pogresan unos, pokusajte ponovo.")


print("Broj elemenata:", len(brojevi))
print("Minimum:", min(brojevi))
print("Maksimum:", max(brojevi))
print("Srednja vrijednost:", sum(brojevi)/len(brojevi))

sorted(brojevi)
print("Sortirana lista:", brojevi)
