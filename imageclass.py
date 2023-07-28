from hopfield import *

arr = image_nodes('one.png')
hop = Hopfield(n = 784, values = arr)
print(hop.values)

