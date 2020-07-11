import tensorflow as tf # tf.__version__

mnist = tf.keras.datasets.mnist # 28x28 images of hand-written digits 0-9

(x_train, y_train), (x_text, y_test) = mnist.load_data()

import matplotlib.pyblot as plt

plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()

print(x_train[0])