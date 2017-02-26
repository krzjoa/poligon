
# Cooked by recipe from: https://www.udemy.com/cluster-analysis-unsupervised-machine-learning-python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class GMM(object):
    '''

    Gaussian Mixture Model

    Parameters
    ----------
    k: int
          Number of gaussian clusters
    max_iter: int
         Maximal number of iterations
    '''

    def __init__(self, k=2, max_iter=20):

        # Number of clusters
        self.k = k
        # Maximal number of iterations to stop algorithm
        self.max_iter = max_iter
        # Mean value of distribution
        self.means = None
        # Covariance matrices
        self.cov_mat = None
        # Responsibilities
        self.R = None


    def fit(self, X):

        n_samples, n_dim  = X.shape

        # Initializing
        self.means = np.zeros((self.k, n_dim))
        self.cov_mat = np.zeros((self.k, n_dim, n_dim))
        self.pi = np.ones(self.k) / self.k
        self.R = np.zeros((N, self.k))

        for k in xrange(self.k):
            # Starting means - randomly chosen points
            self.means[k] = X[np.random.choice(n_samples)]
            self.cov_mat[k] = np.eye(n_dim)

        # Costs in every iteration
        costs = np.zeros(self.max_iter)

        # PDF - probability density function
        # It shows, how strongly a particular sample belongs to the given cluster k
        weighted_pdfs = np.zeros((n_samples, self.k))

        # For loop - maximal iterations number
        for i in xrange(self.max_iter):

            # For every cluster
            for k in xrange(self.k):

                for n in xrange(n_samples):
                    weighted_pdfs[n, k] = self.pi[k] * multivariate_normal.pdf(X[n], self.means[k], self.cov_mat[k])

            # TODO: Simplify
            # Computing responsibilities
            for k in xrange(self.k):
                for n in xrange(n_samples):
                    self.R[n, k] = weighted_pdfs[n, k] / weighted_pdfs[n, :].sum()

            # Computing Gausses params basing on points and their responsibilities
            # TODO: Analyze
            for k in xrange(self.k):
                Nk = self.R[:, k].sum()
                self.pi[k] = Nk / n_samples
                self.means[k] = self.R[:, k].dot(X) / Nk
                self.cov_mat[k] = np.sum(self.R[n, k] * np.outer(X[n] - self.means[k], X[n] - self.means[k]) for n in xrange(N)) / Nk + np.eye(n_dim) * 0.001

            costs[i] = np.log(weighted_pdfs.sum(axis=1)).sum()

            # Stop condition
            if i > 0:
                if np.abs(costs[i] - costs[i-1]) < 0.1:
                    break

        return self

    def plot(self, X):
        random_colors = np.random.random((self.k, 3))
        colors = self.R.dot(random_colors)
        plt.scatter(X[:, 0], X[:, 1], c=colors)
        plt.show()

        print "pi: ", self.pi



if __name__ == "__main__":
    n_dim = 2
    s = 4

    mu1 = np.array([0, 0])
    mu2 = np.array([s, s])
    mu3 = np.array([0, s])

    N = 900
    X = np.zeros((N, n_dim))
    X[:300, :] = np.random.randn(300, n_dim) * mu1
    X[300:600, :] = np.random.randn(300, n_dim) * mu2
    X[600:, :] = np.random.randn(300, n_dim) * mu3

    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()

    K = 3

    # Gaussian Mixture Model
    gmm = GMM(k=K, max_iter=50)

    gmm.fit(X)

    gmm.plot(X)


