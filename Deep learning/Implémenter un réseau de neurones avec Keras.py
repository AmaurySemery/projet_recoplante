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

# Pour utiliser le VGG-16 pré-entraîné