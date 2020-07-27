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