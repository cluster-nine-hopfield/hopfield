from hopfield import *

test = Hopfield(3)
test_copy = Hopfield(3, test.weights, test.values)
assert test.do_synchronous_update() == test_copy.update_nodes([0, 1, 2])
