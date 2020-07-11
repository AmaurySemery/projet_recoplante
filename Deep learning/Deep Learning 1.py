import tensorflow as tf # tf.__version__

mnist = tf.keras.datasets.mnist # 28x28 images of hand-written digits 0-9

(x_train, y_train), (x_text, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_trains, axis=1)
x_test = tf.keras.utils.normalize(x_train, axis=1)

model = tf.keras.models.Sequential()
model.add()

import matplotlib.pyblot as plt

plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()

print(x_train[0])