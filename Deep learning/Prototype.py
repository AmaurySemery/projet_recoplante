# Je commence avec TensorFlow Lite pour faire de la reconnaissance en Deep Learning sur mobile
# Le modèle pré-entraîné que je veux utiliser correspond à "Image classification"
# Sert à identifier des centaines de classes d'objets, notamment des personnes, des animaux, des lieux mais surtout des plantes (ce qui va nous intéresser)
# Mon but va être de faire du "transfer learning", soit de réentraîner le modèle choisi pour lui faire effectuer une autre tâche. Ainsi, je pourrai ajouter de nouvelles catégories d'images.

import tensorflow as tf
assert tf.__version__.startswith('2')

import os
import numpy as np
import matplotlib.pyplot as plt