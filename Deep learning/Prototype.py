# Je commence avec TensorFlow Lite pour faire de la reconnaissance en Deep Learning sur mobile
# Le modèle pré-entraîné que je veux utiliser correspond à "Image classification"
# Sert à identifier des centaines de classes d'objets, notamment des personnes, des animaux, des lieux mais surtout des plantes (ce qui va nous intéresser)
# Mon but va être de faire du "transfer learning", soit de réentraîner le modèle choisi pour lui faire effectuer une autre tâche. Ainsi, je pourrai ajouter de nouvelles catégories d'images.

import tensorflow as tf
assert tf.__version__.startswith('2')

import os
import numpy as np
import matplotlib.pyplot as plt

# Je télécharge la base de données sur les fleurs

_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

zip_file = tf.keras.utils.get_file(origin=_URL,
                                   fname="flower_photos.tgz",
                                   extract=True)

base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

# J'utilise ImageDataGenerator pour redimensionner les images
# Ce faisant, il me faut créer le générateur d'entraînement en spécifiant l'emplacement des données au sortir de l'entraînement (les données sur la plante, la taille de l'image, etc.)
# Ensuite, il me faut créer le générateur de validation avec une approche similaire à la méthode flow_from_directory()

IMAGE_SIZE = 224
BATCH_SIZE = 64

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2)

train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='training')

val_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='validation')

for image_batch, label_batch in train_generator:
  break
image_batch.shape, label_batch.shape

# J'enregistre les étiquettes dans un fichier qui sera téléchargé plus tard

print (train_generator.class_indices)

labels = '\n'.join(sorted(train_generator.class_indices.keys()))

with open('labels.txt', 'w') as f:
  f.write(labels)

!cat labels.txt

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

# Je produits le modèle de base à partir du modèle pré-entraîné MobileNetV2

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                              include_top=False,
                                              weights='imagenet')

# Je gèle la base convolutionnelle créée à partir de l'étape précédente avant de l'utiliser comme extracteur de caractéristiques, puis j'ajoute un classificateur par-dessus.

base_model.trainable = False

# J'ajoute donc une tête de classification

model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(5, activation='softmax')
])

# Je compile le modèle avant de l'entraîner.

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print('Number of trainable variables = {}'.format(len(model.trainable_variables)))

# J'entraîne le modèle.

epochs = 10

history = model.fit(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=epochs,
                    validation_data=val_generator,
                    validation_steps=len(val_generator))

# J'affiche les courbes d'apprentissage.

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
