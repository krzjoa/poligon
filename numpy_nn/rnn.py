# -*- coding: utf-8 -*-

import numpy as np

# RNN-y w numpy

# Źródło: http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/


class RNNNumpy(object):

    def __init__(self, hidden_dim=100, bptt_truncate=4):
        self.hidden_dim = hidden_dim
        self.bppt_truncate = bptt_truncate

    def fit(self, X, y):

        # Inicjalizacja zmiannych w zależności od wymiaru wejścia
        self.U = init_matrix(X.shape[2], self.hidden_dim)
        self.V = init_matrix(self.hidden_dim, X.shape[2])
        self.W = init_matrix(self.hidden_dim, self.hidden_dim)

        # Krok naprzód




    def forward(self, x):
        # Metoda forward obsługuje propagację naprzód jednego
        # przykładu, tj. jednej sekwencji

        # Liczba kroków w czasie
        T = len(x)

        # Podczas propagacji naprzód zapisujemy wszystkie stany ukryte
        # Będą one później podobne podczas różniczkowania przy propagacji
        # wstecznej. Trzymamy ją pod postacią macierzy stanów ukrytych

        S = np.zeros((T + 1, self.hidden_dim))
        #S[-1] = np.zeros(self.hidden_dim) # WTF? Po co to?

        # Macierz wyjścia
        O = np.zeros((T, self.hidden_dim))

        # Iterujemy po każdym kroku w czasie
        for t in xrange(T):
            S[t] = np.tanh(x[t].dot(self.U) + self.W.dot(S[t-1]))
            O[t] = softmax(S[t].dot(self.V))
        return O, S





def init_matrix(n, m):
    n_max = np.max((n, m))
    return np.random.uniform(-np.sqrt(1/n_max), np.sqrt(1/n_max), (n, m))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1)


if __name__ == '__main__':
    rnn = RNNNumpy()

