import numpy as np

# Read the array from disk
new_data = np.loadtxt('test.txt')

# Note that this returned a 2D array!
print(new_data.shape)

# However, going back to 3D is easy if we know the
# original shape of the array
new_data = new_data.reshape((4,5,10))

# Just to check that they're the same...
#assert np.all(new_data == data)
print(new_data)