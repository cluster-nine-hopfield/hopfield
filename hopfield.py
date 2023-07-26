import numpy as np


class Hopfield:
    def __init__(self, n = 3, weights = None, values = None) -> None:
        # n^2 = number of nodes in the network 
        if weights is None:
            self.weights = np.random.choice([-1, 1], size=(n,n))
            self.weights = np.tril(self.weights) + np.tril(self.weights, -1).T # makes the matrix diagonal
            for i in range(n):
                self.weights[n][n] = 0 # nodes don't input to themselves
        else:
            self.weights = np.array(weights) # making sure it's the np object
        if values is None:
            self.values = np.random.choice([-1, 1], size=n)
        else:
            self.values = np.array(values) # making sure it's the np object
        
    def do_instantaneous_update(self):
        node_inputs = self.weights @ self.values
        self.values = self.convert_node_inputs_to_outputs(node_inputs)
        return self.values

    def convert_node_inputs_to_outputs(self, node_inputs):
        outputs = []
        for node_input in node_inputs:
            outputs.append({True : 1, False : -1}[node_input >= 0]) # activation function
        return outputs
    
test = Hopfield(3, [[0, -1, -1], [-1, 0, 1], [-1, 1, 0]], [1, 1, -1])
print(test.do_instantaneous_update())
print(test.do_instantaneous_update())
