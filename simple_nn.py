import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidPrime(x):
    # Pierwsza pochodna z sigmoida
     return np.exp(-x)/((1+ np.exp(-x))**2)






class SimpleNN(object):

    def __init__(self, inp_size, hidden_size, out_size, n_epoch=10):
        self.W1 = np.random.rand(inp_size, hidden_size)
        self.W2 = np.random.rand(hidden_size, out_size)
        self.n_epoch = n_epoch


    def forward(self, X):
        self.z2 = X.dot(self.W1)
        self.a2 = sigmoid(self.z2)

        self.z3 = self.a2.dot(self.W2)
        self.a3 = sigmoid(self.z3)

        return self.a3

    def backward(self):
        pass


    def cost_function_prime(self, X, y):
        # Pierwsza pochodna z funckji kosztu
        # W yHat dostajemy predykcje sieci
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat), sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.W1, delta3)

        delta2 =np.dot(dJdW2, self.W2) * sigmoid(self.z2)
        #dJW1 =

    def fit(self, X, y):

        for i in xrange(self.n_epoch):
            pass






















if __name__ == "__main__":
    pass