import numpy as np

data = np.arange(16).reshape((2,2,2,2))
print(data)
re_data = np.reshape(data, (2, 8))
print(re_data)
print(re_data.shape)