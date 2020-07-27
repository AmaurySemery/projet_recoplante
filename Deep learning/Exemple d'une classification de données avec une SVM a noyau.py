# Documentation :
# https://openclassrooms.com/fr/courses/4470406-utilisez-des-modeles-supervises-non-lineaires/4722466-classifiez-vos-donnees-avec-une-svm-a-noyau
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

import numpy as np

# charger les données

import pandas as pd

data = pd.read_csv('winequality-white.csv', sep=';')


# créer la matrice de données

X = data[data.columns[:-1]].values


# créer le vecteur d'étiquettes

y = data['quality'].values


# transformer en un problème de classification binaire

y_class = np.where(y<6, 0, 1)

# Avant toute chose, nous allons découper nos données en un jeu d'entraînement (X_train, y_train) et un jeu de test (X_test, y_test).

from sklearn import model_selection
X_train, X_test, y_train, y_test = \
    model_selection.train_test_split(X, y_class, test_size=0.3)
    
# Nous pouvons maintenant standardiser les variables, c'est-à-dire les centrer (ramener leur moyenne à 0) et les réduire (ramener leur écart-type à 1), afin qu'elles se placent toutes à peu près sur la même échelle.
    
# standardiser les données

from sklearn import preprocessing

std_scale = preprocessing.StandardScaler().fit(X_train)


X_train_std = std_scale.transform(X_train)

X_test_std = std_scale.transform(X_test)

# OK, nous pouvons enfin entraîner notre première SVM à noyau !

# Créer une SVM avec un noyau gaussien de paramètre gamma=0.01

from sklearn import svm

classifier = svm.SVC(kernel='rbf', gamma=0.01)


# Entraîner la SVM sur le jeu d'entraînement

classifier.fit(X_train_std, y_train)

# Comment se comporte-t-elle sur le jeu de test ? Nous allons pour le comprendre regarder la courbe ROC.

# prédire sur le jeu de test

y_test_pred = classifier.decision_function(X_test_std)


# construire la courbe ROC

from sklearn import metrics

fpr, tpr, thr = metrics.roc_curve(y_test, y_test_pred)


# calculer l'aire sous la courbe ROC

auc = metrics.auc(fpr, tpr)


# créer une figure

from matplotlib import pyplot as plt

fig = plt.figure(figsize=(6, 6))


# afficher la courbe ROC

plt.plot(fpr, tpr, '-', lw=2, label='gamma=0.01, AUC=%.2f' % auc)


# donner un titre aux axes et au graphique

plt.xlabel('False Positive Rate', fontsize=16)

plt.ylabel('True Positive Rate', fontsize=16)

plt.title('SVM ROC Curve', fontsize=16)


# afficher la légende

plt.legend(loc="lower right", fontsize=14)


# afficher l'image

plt.show()

# Problème : le fichier winequality-white.csv n'a pas été trouvé => il faut aller le chercher pour que ça fonctionne. J'avais eu le même problème avec l'image de la plante en noir et blanc, il fallait trouver l'emplacement sur l'ordinateur. 