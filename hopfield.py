import random
import numpy as np 

class Hopfield:
    def __init__(self, n) -> None:
        nodes = np.random.rand(n)