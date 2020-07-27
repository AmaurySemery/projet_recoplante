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