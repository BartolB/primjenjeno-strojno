from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime



BATCH_SIZE = 64
IMG_SIZE = (48, 48)

TRAIN_PATH = r'C:\labos\gtsrb\Train'
TEST_PATH = r'C:\labos\gtsrb\Test'

train_ds = image_dataset_from_directory(
    directory=TRAIN_PATH,
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    validation_split=0.2,
    subset='training',
    seed=123
)

validation_ds = image_dataset_from_directory(
    directory=TRAIN_PATH,
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    validation_split=0.2,
    subset='validation',
    seed=123
)

test_ds = image_dataset_from_directory(
    directory=TEST_PATH,
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False
)

np.save("class_names.npy", train_ds.class_names)



AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)


model = models.Sequential([

    layers.Rescaling(1./255, input_shape=(48, 48, 3)),

    layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu'
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

    # GTSRB = 43 classes
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
    ),

    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
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



plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()


plt.subplot(1, 2, 2)

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()