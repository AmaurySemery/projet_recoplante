# https://openclassrooms.com/fr/courses/4470531-classez-et-segmentez-des-donnees-visuelles/5097666-tp-implementez-votre-premier-reseau-de-neurones-avec-keras
# Implémenter un réseau de neurones avec Keras revient à créer un modèle  Sequential  et à l'enrichir avec les couches correspondantes dans le bon ordre. L'étape la plus difficile est de définir correctement les paramètres de chacune des couches – d'où l'importance de bien comprendre l'architecture du réseau !

from keras.models import Sequential


my_VGG16 = Sequential()  # Création d'un réseau de neurones vide

# A ce stade, vous pouvez déjà implémenter quasiment tout le réseau VGG-16 ! Par exemple, la construction du premier bloc de couches est détaillée ci-dessous :

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D


my_VGG16 = Sequential()  # Création d'un réseau de neurones vide 


# Ajout de la première couche de convolution, suivie d'une couche ReLU

my_VGG16.add(Conv2D(64, (3, 3), input_shape=(224, 224, 3), padding='same', activation='relu'))


# Ajout de la deuxième couche de convolution, suivie  d'une couche ReLU

my_VGG16.add(Conv2D(64, (3, 3), padding='same', activation='relu'))


# Ajout de la première couche de pooling

my_VGG16.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

# J'ai un problème à ce stade : ImportError: Keras requires TensorFlow 2.2 or higher. Install TensorFlow via `pip install tensorflow`
# Pourtant, j'ai mis à jour tensorflow que j'ai à la version 3.1.0

# Ainsi, les trois dernières couches fully-connected et leur fonction d'activation (ReLU pour les deux premières, softmax pour la dernière) sont ajoutées de la manière suivante :

from keras.layers import Flatten, Dense


my_VGG16.add(Flatten())  # Conversion des matrices 3D en vecteur 1D


# Ajout de la première couche fully-connected, suivie d'une couche ReLU

my_VGG16.add(Dense(4096, activation='relu'))


# Ajout de la deuxième couche fully-connected, suivie d'une couche ReLU

my_VGG16.add(Dense(4096, activation='relu'))


# Ajout de la dernière couche fully-connected qui permet de classifier

my_VGG16.add(Dense(1000, activation='softmax'))

# Pour résoudre le problème, je tente la commande 'pip install keras==2.2.4' en suivant les indications sur ce site : https://stackoverflow.com/questions/62482404/error-while-importing-keras-and-tensorflow
# C'est un échec

# Pour utiliser le VGG-16 pré-entraîné :

from keras.applications.vgg16 import VGG16

from keras.layers import Dense


# Charger VGG-16 pré-entraîné sur ImageNet et sans les couches fully-connected

model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))


# Récupérer la sortie de ce réseau

x = model.output


# Ajouter la nouvelle couche fully-connected pour la classification à 10 classes

predictions = Dense(10, activation='softmax')(x)


# Définir le nouveau modèle

new_model = Model(inputs=model.input, outputs=predictions)

# Il y a toujours les messages d'erreurs, mais ça mouline et donne des résultats que je n'arrive pas à analyser

# Première stratégie : on entraîne tout le réseau, donc il faut rendre toutes les couches "entraînables" :

for layer in model.layers:

   layer.trainable = True
   
# Deuxième stratégie : On entraîne seulement le nouveau classifieur et on ne ré-entraîne pas les autres couches :

#[for layer in model.layers:

#   layer.trainable = False]
   
# Troisième stratégie : On entraîne le nouveau classifieur et les couches hautes :

#[for layer in model.layers:

#   layer.trainable = False]

# Il ne reste plus qu'à compiler le nouveau modèle, puis à l'entraîner  :

# Compiler le modèle 

new_model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])


# Entraîner sur les données d'entraînement (X_train, y_train)

model_info = new_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
.
# Résultat : j'ai à la fin une série de messages avec les file qui paraissent défaillants.