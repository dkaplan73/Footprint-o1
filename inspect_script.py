import numpy as np

# Load the .npz file to check its contents
data = np.load("dataset_augmented.npz")

# List all the keys in the dataset
print(data.files)

