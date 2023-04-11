import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('TkAgg')  # set the backend


# Load the dataset
iris = load_iris()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Create an instance of the KNN model
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the model on the training data
knn.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = knn.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Plot the true and predicted labels
plt.scatter(range(len(y_test)), y_test, label='True Labels')
plt.scatter(range(len(y_pred)), y_pred, label='Predicted Labels')
plt.title('True and Predicted Labels')
plt.legend()
plt.show()
