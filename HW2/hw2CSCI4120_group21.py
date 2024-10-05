from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# TODO determine the best k for k-means
# Model our KMeans
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1, 10))
# Fit the data and display to user
visualizer.fit(X)
visualizer.show()

# Get the optimal number of clusters from the visualizer
optimal_k = visualizer.elbow_value_

# Fit KMeans with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
kmeans.fit(X)
y_pred = kmeans.fit_predict(X)

# TODO calculate accuracy for best K
# Calculate the accuracy for using the best K
accuracy = accuracy_score(y_true, y_pred)
# Display accuracy and round to nearest hundreth
print(f"Accuracy for best K ({optimal_k}): {accuracy * 100:.2f}")

# TODO draw a confusion matrix
# set matrix variable as confusion matrix using the built in function from SKlearn
matrix = confusion_matrix(y_true, y_pred)
# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, cmap="Blues", fmt="g")
plt.title(f"Confusion Matrix for K={optimal_k}")
plt.xlabel('Predicted')
plt.ylabel('True')
# Display confusion matrix
plt.show()