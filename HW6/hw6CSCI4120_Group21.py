# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import uniform
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

# use seaborn plotting defaults
import seaborn as sns; sns.set()
from sklearn import datasets, svm, metrics

# The digits dataset
digits = datasets.load_digits()
x = digits.data
y = digits.target

# fit the pca
pca = PCA(n_components=.80)
X_pca = pca.fit_transform(x)

# create the different svm models
svm_linear = svm.SVC(kernel='linear')
svm_rbf = svm.SVC(kernel='rbf')
svm_poly = svm.SVC(kernel='poly')

# params
params = {
    'C': uniform(loc=0., scale=200),
    'gamma': uniform(loc=0., scale=0.2 ),
}
# use the different svm models in a randomized search
rand_search_linear = RandomizedSearchCV(svm_linear, param_distributions=params, n_iter=50, cv=5, n_jobs=-1)
rand_search_rbf = RandomizedSearchCV(svm_rbf, param_distributions=params, n_iter=50, cv=5, n_jobs=-1)
rand_search_poly = RandomizedSearchCV(svm_poly, param_distributions=params, n_iter=50, cv=5, n_jobs=-1)

# fit the different models
rand_search_linear.fit(X_pca, y)
rand_search_rbf.fit(X_pca, y)
rand_search_poly.fit(X_pca, y)

# perform cross validation
linear_score = cross_val_score(rand_search_linear.best_estimator_, X_pca, y, cv=5)
rbf_score = cross_val_score(rand_search_rbf.best_estimator_, X_pca, y, cv=5)
poly_score = cross_val_score(rand_search_poly.best_estimator_, X_pca, y, cv=5)

# display the results
print("The scores for each of the 5 Cross-Validation for the for the linear kernel is: " + str(linear_score))
print("The scores for each of the 5 Cross-Validation for the for the rbf kernel is: " + str(rbf_score))
print("The scores for each of the 5 Cross-Validation for the poly kernel is: " + str(poly_score))

print("The following are going to be the average scores behind the cross validations")
linear_scoreFinal = np.mean(linear_score)
rbf_scoreFinal = np.mean(rbf_score)
poly_scoreFinal = np.mean(poly_score)

print("The score for the linear kernel is: " + str(linear_scoreFinal))
print("The score for the rbf kernel is: " + str(rbf_scoreFinal))
print("The score for the poly kernel is: " + str(poly_scoreFinal))

def plot_digits(data):
    fig, axes = plt.subplots(4, 10, figsize=(10, 4),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8),
                  cmap='binary', interpolation='nearest',
                  clim=(0, 16))
    plt.show()
plot_digits(digits.data)