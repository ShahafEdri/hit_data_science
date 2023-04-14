from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Generate a random 3-dimensional classification problem
X, y = make_classification(n_samples=100, n_features=3, n_redundant=0, n_informative=3,
                           random_state=1, n_clusters_per_class=1, class_sep=0.5)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create an instance of the KNN model
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the model on the training data
knn.fit(X_train, y_train)

# Define the range of values for the x, y and z axes
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
z_min, z_max = X[:, 2].min() - 0.5, X[:, 2].max() + 0.5

# Create a grid of points that covers the entire range of the x, y and z axes
xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1), np.arange(z_min, z_max, 0.1))

# Make predictions on the grid points
Z = knn.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])

# Reshape the predictions to the shape of the xx and yy grids
Z = Z.reshape(xx.shape)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the decision boundary
ax.scatter(xx[Z==0], yy[Z==0], zz[Z==0], marker='.', alpha=0.2, color='blue')
ax.scatter(xx[Z==1], yy[Z==1], zz[Z==1], marker='.', alpha=0.2, color='red')

# Plot the data points
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, alpha=0.8)

# Add axis labels and a title
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title('KNN Classification')

# Show the plot
plt.show()
