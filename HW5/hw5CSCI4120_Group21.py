import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Load the breast cancer dataset
init_data = load_breast_cancer()
X, y = init_data.data, init_data.target

print("Shape of feature matrix X:", X.shape)

# Create and fit a RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X, y)

# Get feature importances
importances = classifier.feature_importances_

# Sort the features by importance
indices = np.argsort(importances)[::-1]

# Select top 2 features based on feature importance
# Lowering the features to 2 got my accuracy / number of features used consistently above .45
bestfeatures = indices[:2]

# Print out the names of the top 5 most important features
top_feature_names = [init_data.feature_names[i] for i in bestfeatures]


# Create a feature matrix with the top features
selectedfeatures = X[:, bestfeatures]

# Perform 5-fold cross-validation using the selected features
scores = cross_val_score(classifier, selectedfeatures, y, cv=5)

# Tune parameters for RandomForestClassifier using GridSearchCV
params = {
    'n_estimators': [10, 50, 100, 200],
    'max_features': ['sqrt', 'log2', 0.1, 0.2, 0.3],
    'max_depth': [1, 2, 5, 8, None],
}

# Perform grid search with a cross validation of 5
grid_search = GridSearchCV(classifier, params, cv=5)
grid_search.fit(selectedfeatures, y)


# Calculate and display average accuracy score for the model with tuned hyperparameters
accuracy = accuracy_score(y, grid_search.predict(selectedfeatures))
print(f"The accuracy on the dataset is: {accuracy:.4f}")

# Calculate the average (accuracy score / number of features)
average_score_per_feature = accuracy / selectedfeatures.shape[1]
print(f"Average for (accuracy score / number of features): {average_score_per_feature:.4f}")

