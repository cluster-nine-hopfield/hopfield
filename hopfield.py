import numpy as np


class Hopfield:
    def __init__(self, n, weights = ) -> None:
        # n^2 = number of nodes in the network 
        self.weights = np.random.choice([-1, 1], size=(n,n))
        self.values = np.random.choice([-1, 1], size=n)

    def do_instantaneous_update(self):
        pass