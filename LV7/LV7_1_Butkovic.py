import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



# MNIST podatkovni skup
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# TODO: prikazi nekoliko slika iz train skupa
plt.figure(figsize=(5, 4))

for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.axis("off")


plt.show()


# Skaliranje vrijednosti piksela na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# Slike 28x28 piksela se predstavljaju vektorom od 784 elementa
x_train_s = x_train_s.reshape(60000, 784)
x_test_s = x_test_s.reshape(10000, 784)

# Kodiraj labele (0, 1, ... 9) one hot encoding-om
y_train_s = keras.utils.to_categorical(y_train, 10)
y_test_s = keras.utils.to_categorical(y_test, 10)


# TODO: kreiraj mrezu pomocu keras.Sequential(); prikazi njenu strukturu pomocu .summary()
model = Sequential()
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.summary()    


# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy'])


# TODO: provedi treniranje mreze pomocu .fit()
model.fit(x_train_s, y_train_s, epochs=5, batch_size=128)


# TODO: Izracunajte tocnost mreze na skupu podataka za ucenje i skupu podataka za testiranje
train_loss, train_acc = model.evaluate(x_train_s, y_train_s, verbose=0)
test_loss, test_acc = model.evaluate(x_test_s, y_test_s, verbose=0)

print(f"Tocnost na train skupu: {train_acc:.4f}")
print(f"Tocnost na test skupu: {test_acc:.4f}")

# TODO: Prikazite matricu zabune na skupu podataka za testiranje
y_pred_probs = model.predict(x_test_s)
y_pred = np.argmax(y_pred_probs, axis=1)
cm = confusion_matrix(y_test, y_pred)


disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Matrica zabune")
plt.show()



# TODO: Prikazi nekoliko primjera iz testnog skupa podataka koje je izgrađena mreza pogresno klasificirala
wrong = np.where(y_pred != y_test)[0]

plt.figure(figsize=(12, 8))

for i in range(10):
    idx = wrong[i]

    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[idx], cmap="gray")
    plt.title(f"Stvarno: {y_test[idx]}\nPredikcija: {y_pred[idx]}")
    plt.axis("off")

plt.tight_layout()
plt.show()

