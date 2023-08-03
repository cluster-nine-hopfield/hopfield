from hopfield import *

network = Hopfield.from_bw_image("hello_there.png")
initial_values = np.array(network.values)
network.save_as_image()
for i in range(250):
    network.perturb(1)
    if i % 10 == 0:
        network.save_as_image()

i = 0
while network.hamming_distance(initial_values, network.values) > 10:
    # network.do_random_update()
    network.do_synchronous_update()
    print("did random update")
    if i % 20 == 0:
        network.save_as_image()
        print("saved as image")
    i += 1

network.save_as_image()
network.animate(True)
print(i)