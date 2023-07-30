import os
import imageio
import numpy as np
from PIL import Image
from numpy import asarray


# Hopfield network
class Hopfield:
    # n = number of nodes in the network
    # weights = n x n matrix of weights
    # values = n x 1 vector of values
    def __init__(self, shape=(5, 5), weights=None, values=None) -> None:
        n = shape[0] * shape[1]
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
        self.shape = shape
        self.images_created_from_this_class = []

    def do_synchronous_update(self):
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

    def update_node(self, node_index):
        node_input = self.values @ self.weights[node_index]
        self.values[node_index] = self.convert_node_inputs_to_outputs(node_input)[0]
        return self.values[node_index]

    def update_nodes(self, list_of_node_indexes):
        node_inputs = self.values @ [
            self.weights[node_index] for node_index in list_of_node_indexes
        ]
        node_outputs = self.convert_node_inputs_to_outputs(node_inputs)
        for i in range(len(list_of_node_indexes)):
            self.values[list_of_node_indexes[i]] = node_outputs[i]
        return node_outputs

    def convert_node_inputs_to_outputs(self, node_inputs):
        outputs = []
        for node_input in node_inputs:
            outputs.append({True: 1, False: -1}[node_input >= 0])  # activation function
        return np.array(outputs)

    def is_steady(self):
        new = Hopfield(self.weights, self.values, self.shape)
        new.do_synchronous_update()
        return np.array_equal(self.values, new.values)    

    def convert_values_to_image(self):
        vals = self.values.size
        rectangle = np.reshape(self.values, self.shape)
        rectangle = ((rectangle * -1 + 1)/2 * 255).astype(np.uint8)
        print(rectangle)
        img = Image.fromarray(rectangle)
        return img

    def display(self):
        img = self.convert_values_to_image()
        img.show()

    def save_as_image(self):
        img = self.convert_values_to_image()
        i = 0
        while os.path.exists("network" + str(i) + ".png"):
            i += 1
        img.save("network" + str(i) + ".png")
        self.images_created_from_this_class.append("network" + str(i) + ".png")
        
    def animate(self, delete_images_afterwards=False):
        images = [imageio.imread(f) for f in self.images_created_from_this_class]
        i = 0
        while os.path.exists("network" + str(i) + ".gif"):
            i += 1
        imageio.mimwrite("network" + str(i) + ".gif", images, duration=2000)
        if delete_images_afterwards:
            for image in self.images_created_from_this_class:
                os.remove(image)

    def train_on_values(self):
        for i in range(self.n):
            for j in range(self.n):
                self.weights[j][i] = self.weights[i][j] = (i!=j) * self.values[i] * self.values[j]

    def perturb(self, num, replace=True):
        indexes_to_flip = np.random.choice(list(range(self.n)), size=num, replace=replace)
        for i in indexes_to_flip:
            self.values[i] *= -1
    
    def flip_values(self):
        for (i, value) in enumerate(self.values):
            self.values[i] = -value

    @classmethod
    def from_image(cls, img):
        shape, values = cls.convert_image_to_values(img)
        weights = cls.generate_weights_from_values(values)
        return cls(shape, weights, values)

    @staticmethod
    def convert_image_to_values(img):
        image = Image.open(img)
        image_as_array = asarray(image)
        image_array = None
        if type(image_as_array[0][0]) == np.uint8:
            image_array = np.array([[1 if pixel == 0 else -1 for pixel in row] for row in asarray(image_as_array)])
        elif type(image_as_array[0][0]) == np.ndarray:
            image_array = np.array([[1 if pixel[0] == 0 else -1 for pixel in row] for row in asarray(image_as_array)])

        return (image_array.shape, image_array.flatten())

    @staticmethod
    def generate_weights_from_values(values):
        weights = np.zeros(shape=(len(values), len(values)))
        for i in range(len(values)):
            for j in range(len(values)):
                weights[i][j] = (values[i] != values[j]) * values[i] * values[j]
        return weights

    @staticmethod
    def hamming_distance(values1, values2):
        return sum(values1 != values2)
