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
    def __init__(
        self, shape=(5, 5), weights=None, values=None, folder_to_save_to=None
    ) -> None:
        self.perturbed_nodes = None
        n = shape[0] * shape[1]
        if weights is None:
            self.weights = np.zeros((n,n))
            self.weights = (
                np.tril(self.weights) + np.tril(self.weights, -1).T
            )  # makes the matrix diagonal
            for i in range(n):
                self.weights[i][i] = 0.0  # nodes don't input to themselves
        else:
            self.weights = np.array(weights)  # making sure it's the np object
        if values is None:
            self.values = np.random.choice([-1.0, 1.0], size=n)
        else:
            self.values = np.array(values)  # making sure it's the np object
        self.n = n
        self.shape = shape
        self.images_created_from_this_class = []
        self.size = self.values.size

        if folder_to_save_to is None:
            i = 0
            while os.path.isdir("network" + str(i)):
                i += 1
            self.folder_to_save_to = "network" + str(i)
        else:
            self.folder_to_save_to = folder_to_save_to

    def do_synchronous_update(self):
        node_inputs = self.weights @ self.values
        # print("node_inputs: " + str(node_inputs))
        self.values = self.convert_node_inputs_to_outputs(node_inputs)
        return self.values

    def do_asynchronous_update(self):
        for node in range(self.n):
            self.update_node(node)

    def do_random_update(self):
        index_of_node_to_update = np.random.randint(0, self.n)
        node_input = self.values @ self.weights[index_of_node_to_update]
        self.values[index_of_node_to_update] = self.convert_node_inputs_to_outputs(
            [node_input]
        )[0]
        return self.values[index_of_node_to_update]

    def update_node(self, node_index):
        node_input = self.values @ self.weights[node_index]
        self.values[node_index] = self.convert_node_inputs_to_outputs(node_input)
        return self.values[node_index]

    def update_nodes(self, list_of_node_indexes):
        # updates all these nodes at the same time
        node_inputs = self.values @ [
            self.weights[node_index] for node_index in list_of_node_indexes
        ]
        node_outputs = self.convert_node_inputs_to_outputs(node_inputs)
        for i in range(len(list_of_node_indexes)):
            self.values[list_of_node_indexes[i]] = node_outputs[i]
        return node_outputs

    def convert_node_inputs_to_outputs(self, node_inputs):
        if isinstance(node_inputs, np.float64):
            return {True: 1.0, False: -1.0}[node_inputs >= 0]
        outputs = []
        for node_input in node_inputs:
            outputs.append({True: 1.0, False: -1.0}[node_input >= 0])  # activation function
        return np.array(outputs)

    def is_steady(self):
        new = Hopfield(self.shape, self.weights, self.values)
        new.do_synchronous_update()
        return np.array_equal(self.values, new.values)

    def convert_values_to_image(self):
        rectangle = np.reshape(self.values, self.shape)
        rectangle = ((rectangle * -1 + 1) / 2 * 255).astype(np.uint8)
        print(rectangle)
        img = Image.fromarray(rectangle)
        return img

    def display(self):
        img = self.convert_values_to_image()
        img.show()

    def save_as_image(self):
        if not os.path.isdir(self.folder_to_save_to):
            os.mkdir(self.folder_to_save_to)
        img = self.convert_values_to_image()
        i = 0
        while os.path.exists(self.folder_to_save_to + "/network" + str(i) + ".png"):
            i += 1
        img.save(self.folder_to_save_to + "/network" + str(i) + ".png")
        self.images_created_from_this_class.append(
            self.folder_to_save_to + "/network" + str(i) + ".png"
        )

    def animate(self, delete_images_afterwards=False):
        if not os.path.isdir(self.folder_to_save_to):
            os.mkdir("network" + str(i))
        images = [imageio.imread(f) for f in self.images_created_from_this_class]
        i = 0
        while os.path.exists(self.folder_to_save_to + "/network" + str(i) + ".gif"):
            i += 1
        imageio.mimwrite(
            self.folder_to_save_to + "/network" + str(i) + ".gif",
            images,
            duration=0.000286000286,
        )
        print("duration:")
        print(1000 / len(images))
        if delete_images_afterwards:
            for image in self.images_created_from_this_class:
                os.remove(image)

    def train_on_values(self, storkey = False):
        if not storkey:
            self.weights = self.generate_weights_from_values(self.values, self.size)
        else:
            self.storkey_train(self.values)
    
    def storkey_train(self, array):

        K = np.zeros((self.size,self.size))
        
        for i in range(self.size):
            for j in range(self.size):
                
                W_i = self.weights[i,:]
                M = np.multiply(W_i, array)
                #M = np.delete(M,[i,j])
                M[i] = 0
                M[j] = 0

                K[i][j] = np.sum(M)


        for i in range(self.size):
            for j in range(self.size):

                h_ij = K[i][j]
                h_ji = K[j][i]

                e_i = array[i]
                e_j = array[j]

                term = e_i * e_j - e_i * h_ji - e_j * h_ij
                term = (1.0/self.size) * term

                self.weights[i][j] += term 

        np.fill_diagonal(self.weights,0)

        # W_i = self.weights
        # M = np.outer(W_i, array)
        # np.fill_diagonal(M, 0)

        # K = M.sum(axis=1)
        # K = K[:, np.newaxis]

        # term = np.einsum('i,j->ij', array, array) - np.multiply(array, K) - np.multiply(K, array.T)
        # term *= 1.0 / self.size

        # self.weights += term
        
        

     

    def perturb(self, num, replace=False):
        indexes_to_flip = np.random.choice(
            list(np.arange(self.n)), size=num, replace=replace
        )
        for i in indexes_to_flip:
            self.values[i] *= -1

    def flip_values(self):
        for i, value in enumerate(self.values):
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
        if isinstance(image_as_array[0][0], np.uint8):
            image_array = np.array(
                [
                    [1.0 if pixel == 0 else -1.0 for pixel in row]
                    for row in asarray(image_as_array)
                ]
            )
        elif isinstance(image_as_array[0][0], np.ndarray):
            image_array = np.array(
                [
                    [1.0 if (pixel[-1] == 255) else -1.0 for pixel in row]
                    for row in asarray(image_as_array)
                ]
            )
        elif isinstance(image_as_array[0][0], np.bool_):
            image_array = np.array([[1 * (not pixel) for pixel in row] for row in image_as_array])
        return (image_array.shape, image_array.flatten())

    @staticmethod
    def generate_weights_from_values(values, size):

        
        weights = np.outer(values, values) * (1.0 / size)
        np.fill_diagonal(weights, 0)
        return weights

    @staticmethod
    def hamming_distance(values1, values2):
        return sum(values1 != values2)
    


    def train_on_image(self, img):
        values = self.convert_image_to_values(img)[1]
        weights = self.generate_weights_from_values(values, self.size)
        self.weights += weights

    def sync_update_until_steady(self, infinite = False):
        count = 0
        iterations = []
        if infinite == False:
            for count in range(10):
                self.do_synchronous_update()
                #print(self.values)
                count += 1
                iterations.append(self.energy_function())
                if self.is_steady():
                    break
            
            return [iterations,count]

        else:
            while(self.is_steady() != True):
                self.do_synchronous_update()
                count+=1
                iterations.append(list(self.values))

            return [iterations, count]

    def train_on_new(self, values):
        weights = self.generate_weights_from_values(values, self.size)
        self.weights += weights

    def async_update_until_steady(self):
        count = 0
        while not self.is_steady():
            self.do_asynchronous_update()
            count += 1
        return count

    def save_perturbed(self):
        # save the perturbed nodes in a list
        self.perturbed_nodes = np.array(self.values)
        return self.perturbed_nodes

    def restore_perturbed(self):
        # restore the perturbed nodes from the list
        self.values = self.perturbed_nodes

    def generate_random_image(self):
        return np.random.choice([-1.0, 1.0], size=self.n)
    

    def generate_random_image(self):
        return np.random.choice([-1, 1], size=self.n)
    
    def energy_function(self):
        sum = 0.0
        for i in range(self.size):
            for j in range(self.size):
                sum += self.weights[i][j] * self.values[i] * self.values[j]

        return sum * -0.5