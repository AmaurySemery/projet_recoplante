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

# Jusque là, on n'a fait qu'entraîner quelques couches au-dessus d'un modèle de base MobileNet V2. Les neurones du réseau pré-entraîné n'ont donc pas été mis à jour.
# Pour augmenter davantage les performances, on peut entraîner les neurones du modèle pré-entraîné parallèlement à la formation des classements qu'on a ajouté. Le processus de formation forcera les pondérations à être ajustées à partir de cartes d'entités génériques vers des entités associées spécifiquement à notre base de données.

# Pour se faire, il faut déverrouiller les couches initiales du modèle.
# Pour se faire, il suffit de dégeler le modèle "base_model" et de définir les couches inférieures de manière à ne pas pouvoir être entraînées. Ensuite, on recompile le modèle (nécessaire pour que les modifications prennent effet), puis on reprend l'entraînement.

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False

# On compile le modèle en utilisant un taux d'entraînement plus faible.

model.compile(loss='categorical_crossentropy',
              optimizer = tf.keras.optimizers.Adam(1e-5),
              metrics=['accuracy'])

model.summary()

print('Number of trainable variables = {}'.format(len(model.trainable_variables)))

# Puis on continue l'entraînement du modèle.

history_fine = model.fit(train_generator,
                         steps_per_epoch=len(train_generator),
                         epochs=5,
                         validation_data=val_generator,
                         validation_steps=len(val_generator))

# On convertir en TFLite
# Pour ça, on enregistre le modèle à l'aide de tf.saved_model.save, puis on convertit le modèle enregistré dans un format compatible avec TFLite.

saved_model_dir = 'save/fine_tuning'
tf.saved_model.save(model, saved_model_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

# On télécharge le modèle converti avec les étiquettes.

from google.colab import files

files.download('model.tflite')
files.download('labels.txt')

# Après quoi, on peut jeter un oeil aux courbes d'apprentissage et noter la différence.

acc = history_fine.history['accuracy']
val_acc = history_fine.history['val_accuracy']

loss = history_fine.history['loss']
val_loss = history_fine.history['val_loss']

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

# Using a pre-trained model for feature extraction: When working with a small dataset, it is common to take advantage of features learned by a model trained on a larger dataset in the same domain. This is done by instantiating the pre-trained model and adding a fully-connected classifier on top. The pre-trained model is "frozen" and only the weights of the classifier get updated during training. In this case, the convolutional base extracted all the features associated with each image and you just trained a classifier that determines the image class given that set of extracted features.

#Fine-tuning a pre-trained model: To further improve performance, one might want to repurpose the top-level layers of the pre-trained models to the new dataset via fine-tuning. In this case, you tuned your weights such that your model learned high-level features specific to the dataset. This technique is usually recommended when the training dataset is large and very similar to the orginial dataset that the pre-trained model was trained on.
