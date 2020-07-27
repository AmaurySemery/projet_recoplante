# Implémenter un réseau de neurones avec Keras revient à créer un modèle  Sequential  et à l'enrichir avec les couches correspondantes dans le bon ordre. L'étape la plus difficile est de définir correctement les paramètres de chacune des couches – d'où l'importance de bien comprendre l'architecture du réseau !

from keras.models import Sequential


my_VGG16 = Sequential()  # Création d'un réseau de neurones vide