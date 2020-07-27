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

# Par exemple :

# Charger l'image sous forme d'une matrice de pixels

img = np.array(Image.open('/home/popschool/Documents/GitHub/projet_recoplante/Images_test/bruyere_des_marais_NB.jpg'))


# Générer le bruit gaussien de moyenne nulle et d'écart-type 7 (variance 49)

noise = np.random.normal(0, 7, img.shape)


# Créer l'image bruitée et l'afficher

noisy_img = Image.fromarray(img + noise).convert('L')

noisy_img.show()

from PIL import ImageFilter


# Appliquer le lissage par moyennage (fenêtre de taille 9) et afficher le résultat

noisy_img.filter(ImageFilter.BoxBlur(1)).show()