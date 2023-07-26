import numpy as np


# Hopfield network
class Hopfield:
    # n = number of nodes in the network
    # weights = n x n matrix of weights
    # values = n x 1 vector of values
    def __init__(self, n=3, weights=None, values=None) -> None:
        if weights is None:
            self.weights = np.random.choice([-1, 1], size=(n, n))
            self.weights = (
                np.tril(self.weights) + np.tril(self.weights, -1).T
            )  # makes the matrix diagonal
            for i in range(n):
                self.weights[i][i] = 0  # nodes don't input to themselves
        else:
            self.weights = np.array(weights)  # making sure it's the np object
        if values is None:
            self.values = np.random.choice([-1, 1], size=n)
        else:
            self.values = np.array(values)  # making sure it's the np object
        self.n = n

    def do_instantaneous_update(self):
        node_inputs = self.weights @ self.values
        self.values = self.convert_node_inputs_to_outputs(node_inputs)
        return self.values

    def do_random_update(self):
        index_of_node_to_update = np.random.randint(0, self.n)
        node_input = self.values @ self.weights[index_of_node_to_update]
        self.values[index_of_node_to_update] = self.convert_node_inputs_to_outputs(
            node_input
        )[0]
        return self.values[index_of_node_to_update]

    def convert_node_inputs_to_outputs(self, node_inputs):
        outputs = []
        for node_input in node_inputs:
            outputs.append({True: 1, False: -1}[node_input >= 0])  # activation function
        return outputs


test = Hopfield(3)
print(test.do_instantaneous_update())
print(test.do_instantaneous_update())
