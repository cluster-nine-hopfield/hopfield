from hopfield import *


# test that updating all the nodes at the same time is the same as synchronous update
# test = Hopfield(3)
# test_copy = Hopfield(3, test.weights, test.values)
# assert test.do_synchronous_update() == test_copy.update_nodes([0, 1, 2])


# test initialization from image
test = Hopfield.from_bw_image("H.png")
test.train_on_image("E.png")
# test.save_as_image()
# test.perturb(100)
test.perturb(10, False)
#test.save_as_image()
test.save_perturbed()
#sync until steady
print(test.sync_update_until_steady())
test.restore_perturbed()
print(test.async_update_until_steady())

# print(test.values)
# print(test.values.shape)
# print(test.values.size)
# print(test.weights.shape)
# print(test.weights.size)
# print(test.is_steady())
