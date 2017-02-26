#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ======= LOADING DATA ===========
import pandas as pd

df = pd.read_csv('../data/student_height_weight.txt', sep='\t')

# ========== COMPUTATIONS =========
import numpy as np

X = df.values[:, 0]
y = df.values[:, 1].T

bias = np.ones((len(X), ))

X = np.vstack((bias, X)).T

# b*X + e = y

#print X
# Zmienne skorelowane są na pewno niezależne
# Zmnienne mogą być nieskorelowane i zależne, gdyż korelacja określa jedynie zależność liniową
# Zmienna n

# Obliczanie macierzy kowariancji
cov = X.T.dot(X) / len(X)

print cov








