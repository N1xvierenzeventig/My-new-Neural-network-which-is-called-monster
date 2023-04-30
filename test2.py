import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

data = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf. keras.utils.normalize(x_test, axis=1)

Neural_Network = tf.keras.models.Sequential()
Neural_Network.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
Neural_Network.add(tf.keras.layers.Dense(256, activation="relu"))
Neural_Network.add(tf.keras.layers.Dense(256, activation="relu"))
Neural_Network.add(tf.keras.layers.Dense(10, activation="softmax"))

Neural_Network.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

Neural_Network.fit(x_train, y_train, epochs=25)

Neural_Network.save("Monster")

Neural_Network = tf.keras.models.load_model("Monster")

image_number = 1
while os.path.isfile(f"Project/Digit{image_number}.png"):
   try:
      img = cv2.imread(f"Project/Digit{image_number}.png")[:,:,0]
      img = np.invert(np.array([img]))
      prediction = Neural_Network.predict(img)
      print(f"Your digit is most probably a {np.argmax(prediction)}")
      plt.imshow(img[0], cmap=plt.cm.binary)
      plt.show()
   except:
      print("Error!")
   finally:
      image_number += 1

