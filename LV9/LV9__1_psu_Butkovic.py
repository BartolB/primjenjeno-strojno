from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime




train_ds = image_dataset_from_directory(
    directory='GTSRB/Train',
    labels='inferred',
    label_mode='categorical',
    batch_size=64,
    image_size=(48, 48),
    validation_split=0.2,
    subset='training',
    seed=123
)

validation_ds = image_dataset_from_directory(
    directory='GTSRB/Train',
    labels='inferred',
    label_mode='categorical',
    batch_size=64,
    image_size=(48, 48),
    validation_split=0.2,
    subset='validation',
    seed=123
)

test_ds = image_dataset_from_directory(
    directory='GTSRB/Test',
    labels='inferred',
    label_mode='categorical',
    batch_size=64,
    image_size=(48, 48),
    shuffle=False
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)




model = models.Sequential([

    layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu',
        input_shape=(48, 48, 3)
    ),

    layers.MaxPooling2D(pool_size=(2, 2)),


    layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu'
    ),

    layers.MaxPooling2D(pool_size=(2, 2)),


    layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='relu'
    ),

    layers.MaxPooling2D(pool_size=(2, 2)),


    layers.Flatten(),

    layers.Dense(512, activation='relu'),

    layers.Dropout(0.5),

    layers.Dense(43, activation='softmax')
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
        log_dir=log_dir,
        update_freq=100
    ),

    keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
]



history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=15,
    callbacks=my_callbacks
)




best_model = keras.models.load_model('best_model.keras')

print("\nNajbolji model uspješno učitan.\n")



y_true = np.concatenate([
    np.argmax(y, axis=1)
    for x, y in test_ds
])

predictions = best_model.predict(test_ds)

y_pred = np.argmax(predictions, axis=1)




test_acc = accuracy_score(y_true, y_pred)

print(f"Točnost na testnom skupu: {test_acc:.4f}")




cm_test = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(14, 12))

sns.heatmap(
    cm_test,
    cmap='Blues'
)

plt.title("Matrica zabune - testni skup")
plt.xlabel("Predviđena klasa")
plt.ylabel("Stvarna klasa")

plt.show()