import numpy as np
import matplotlib.pyplot as plt
import os

# Get the parent directory (Project3)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, 'data', 'processed')

# Load processed data
X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
y_train = np.load(os.path.join(data_dir, 'y_train.npy'))

# Display first 9 images (faces and non-faces)
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap='gray')
    label = "FACE" if y_train[i] == 1 else "NON-FACE"
    ax.set_title(label)
    ax.axis('off')

plt.tight_layout()
plt.show()