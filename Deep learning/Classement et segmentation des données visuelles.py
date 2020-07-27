# https://openclassrooms.com/fr/courses/4470531-classez-et-segmentez-des-donnees-visuelles/5024566-appliquez-vos-premiers-traitements-dimages

import matplotlib.pyplot as plt

import numpy as np

from PIL import Image


# Charger l'image comme matrice de pixels

img = np.array(Image.open('/home/popschool/Documents/GitHub/projet_recoplante/Images_test/bruyere_des_marais_NB.jpg'))


# Générer et afficher l'histogramme

# Pour le normaliser : argument density=True dans plt.hist

# Pour avoir l'histogramme cumulé : argument cumulative=True

n, bins, patches = plt.hist(img.flatten(), bins=range(256))

plt.show()

# à gauche => pixels noirs ; au milieu => nuances de gris ; à droite => pixels blancs

# Voir la documentation : https://pillow.readthedocs.io/en/3.1.x/reference/Image.html pour utiliser la librairie