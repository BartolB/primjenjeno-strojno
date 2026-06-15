from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model = keras.models.load_model('best_model.keras')

print("Model uspješno učitan.")



class_names = np.load("class_names.npy", allow_pickle=True)

image_path = r'C:\labos\test_slika1.jpg'

img = tf.keras.utils.load_img(
    image_path,
    target_size=(48, 48)
)

img_array = tf.keras.utils.img_to_array(img)


img_array = img_array / 255.0

img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)

predicted_class = np.argmax(predictions)

confidence = np.max(predictions)


print(f"Predviđena klasa: {predicted_class}")
print(f"Znak: {class_names[predicted_class]}")
print(f"Pouzdanost: {confidence:.4f}")

plt.imshow(img)
plt.title(f"{class_names[predicted_class]}")
plt.axis('off')
plt.show()