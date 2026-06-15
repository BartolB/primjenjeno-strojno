
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime



(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train_s = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test_s = x_test.reshape(-1, 28, 28, 1) / 255.0


y_train_s = to_categorical(y_train, num_classes=10)
y_test_s = to_categorical(y_test, num_classes=10)



model = models.Sequential([

  
    layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='valid',
        activation='relu',
        input_shape=(28, 28, 1)
    ),


    layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=2
    ),

  
    layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='valid',
        activation='relu'
    ),


    layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=2
    ),

   
    layers.Flatten(),

 
    layers.Dense(64, activation='relu'),

  
    layers.Dense(10, activation='softmax')
])


model.summary()


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)



log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

my_callbacks = [

    keras.callbacks.TensorBoard(
        log_dir='logs',
        update_freq=100
    ),

    keras.callbacks.ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )

]



history = model.fit(
    x_train_s,
    y_train_s,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    callbacks=my_callbacks
)


best_model = keras.models.load_model('best_model.h5')

print("\nNajbolji model uspješno učitan.\n")


# Predikcije
y_train_pred = np.argmax(best_model.predict(x_train_s), axis=1)
y_test_pred = np.argmax(best_model.predict(x_test_s), axis=1)

# Točnost
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"Točnost na skupu za učenje: {train_acc:.4f}")
print(f"Točnost na testnom skupu: {test_acc:.4f}")


cm_train = confusion_matrix(y_train, y_train_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues')
plt.title("Matrica zabune - skup za učenje")
plt.xlabel("Predviđena klasa")
plt.ylabel("Stvarna klasa")
plt.show()



cm_test = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens')
plt.title("Matrica zabune - testni skup")
plt.xlabel("Predviđena klasa")
plt.ylabel("Stvarna klasa")
plt.show()
