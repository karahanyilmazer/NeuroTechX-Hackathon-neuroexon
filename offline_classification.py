# %%
#!%matplotlib qt
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix as cm

# %%


def fisher_rank(X, y):

    n_feat = X.shape[1]
    c1 = X[y == np.unique(y)[0]]
    c2 = X[y == np.unique(y)[1]]
    scores = np.zeros(n_feat)

    for i in range(n_feat):
        scores[i] = (np.mean(c1[:, i]) - np.mean(c2[:, i]))**2 / (
            np.var(c1[:, i]) + np.var(c2[:, i]))

    ranks = scores.argsort()[::-1]
    ranks = list(ranks)

    return scores, ranks


def plot_acc_for_n_feats(train_acc, cv_acc, method='fisher'):
    train_mean = [np.mean(acc) * 100 for acc in train_acc]
    train_std = [np.std(acc) * 100 for acc in train_acc]
    train_lower = [mean - std for mean, std in zip(train_mean, train_std)]
    train_upper = [mean + std for mean, std in zip(train_mean, train_std)]

    cv_mean = [np.mean(acc) * 100 for acc in cv_acc]
    cv_std = [np.std(acc) * 100 for acc in cv_acc]
    cv_lower = [mean - std for mean, std in zip(cv_mean, cv_std)]
    cv_upper = [mean + std for mean, std in zip(cv_mean, cv_std)]

    x = np.arange(1, len(train_acc) + 1)

    if method == 'fisher':
        title = 'Fisher Ranking'
        xlabel = 'Number of Features'
    elif method == 'pca':
        title = 'PCA'
        xlabel = 'Number of Components'

    plt.figure()
    plt.plot(x, train_mean, label='Training ACC')
    plt.fill_between(x, train_lower, train_upper, alpha=0.1, color='blue')
    plt.plot(x, cv_mean, label='CV ACC')
    plt.fill_between(x, cv_lower, cv_upper, alpha=0.1, color='orange')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Accuracy (in %)')
    plt.xlim(1, n_tot_feats)
    plt.xticks(np.arange(1, n_tot_feats + 1))
    plt.legend()
    plt.grid()

    plt.show()


def plot_conf_mat(cm_train, cm_test, labels=[]):

    # Create the subplots
    _, axs = plt.subplots(1, 2)
    cmap = 'magma'

    # Plot the confusion matrices
    sns.heatmap(cm_train, annot=True, ax=axs[0], cbar=False, cmap=cmap)
    sns.heatmap(cm_test,
                annot=True,
                ax=axs[1],
                cbar=False,
                yticklabels=False,
                cmap=cmap)

    axs[0].set_title('Training Set')
    axs[0].set_xlabel('Predicted label')
    axs[0].set_ylabel('True label')
    axs[0].xaxis.set_ticklabels(labels)
    axs[0].yaxis.set_ticklabels(labels)
    axs[0].tick_params(axis='both',
                       which='both',
                       bottom=False,
                       left=False,
                       right=False,
                       top=False)

    axs[1].set_title('Test Set')
    axs[1].set_xlabel('Predicted label')
    axs[1].xaxis.set_ticklabels(labels)
    axs[1].tick_params(axis='both',
                       which='both',
                       bottom=False,
                       left=False,
                       right=False,
                       top=False)

    plt.suptitle(f'Confusion Matrices')
    plt.show()


# %%
file = os.path.join('pickles', 'Xy_train.pkl')
with open(file, 'rb') as pkl_file:
    X_train, y_train = pickle.load(pkl_file)

file = os.path.join('pickles', 'Xy_test.pkl')
with open(file, 'rb') as pkl_file:
    X_test, y_test = pickle.load(pkl_file)

# Get the total number of features
n_tot_feats = X_train.shape[1]

# Initialize the scaler
ss = StandardScaler()

# Initialize the normalizer
normalizer = Normalizer()

# Initialize the classifier
svc = SVC(class_weight='balanced', probability=True, random_state=1001)
lda = LinearDiscriminantAnalysis()

# Initialize the repeated stratified k-fold CV
n_splits, n_repeats = 5, 3
rskf = RepeatedStratifiedKFold(n_splits=n_splits,
                               n_repeats=n_repeats,
                               random_state=1001)

# %%
feat_train_acc = []
feat_cv_acc = []

for n_pca_feats in range(1, n_tot_feats + 1):

    train_acc = []
    cv_acc = []

    # Iterate over the stratified CV fold
    for train_idx, test_idx in rskf.split(X_train, y_train):

        # Get the training data of the current fold
        X_train_curr = X_train[train_idx]
        y_train_curr = y_train[train_idx]

        # Get the test data of the current fold
        X_test_curr = X_train[test_idx]
        y_test_curr = y_train[test_idx]

        # Initialize the PCA
        pca = PCA(n_pca_feats)

        # Select the pipeline
        pipe = Pipeline([('Normalizer', normalizer), ('PCA', pca),
                         ('SVC', svc)])
        pipe = Pipeline([('Standard Scaler', ss), ('PCA', pca), ('LDA', lda)])
        pipe = Pipeline([('Standard Scaler', ss), ('PCA', pca), ('SVC', svc)])

        # Fit the pipeline to the training data
        pipe.fit(X_train_curr, y_train_curr)

        # Append the training and CV accuracies to the list
        train_acc.append(pipe.score(X_train_curr, y_train_curr))
        cv_acc.append(pipe.score(X_test_curr, y_test_curr))

    feat_train_acc.append(train_acc)
    feat_cv_acc.append(cv_acc)

plot_acc_for_n_feats(feat_train_acc, feat_cv_acc, method='pca')
# RESULT: Use 13 features

# %%
# Define the number of PCA components
n_pca_feats = 8

# Initialize the PCA
pca = PCA(n_pca_feats)

# Define the pipeline
pipe = Pipeline([('Standard Scaler', ss), ('PCA', pca), ('LDA', lda)])
pipe = Pipeline([('Standard Scaler', ss), ('PCA', pca), ('SVC', svc)])

# Initialize a list for the accuracies
train_acc = []
cv_acc = []

# Iterate over the stratified CV fold
for train_idx, test_idx in rskf.split(X_train, y_train):

    # Get the training data of the current fold
    X_train_curr = X_train[train_idx]
    y_train_curr = y_train[train_idx]

    # Get the test data of the current fold
    X_test_curr = X_train[test_idx]
    y_test_curr = y_train[test_idx]

    # Fit the pipeline to the training data
    pipe.fit(X_train_curr, y_train_curr)

    # Append the training and CV accuracies to the list
    train_acc.append(pipe.score(X_train_curr, y_train_curr))
    cv_acc.append(pipe.score(X_test_curr, y_test_curr))

# Calculate the mean and STD of the accuracies
train_mean = np.mean(train_acc)
train_std = np.std(train_acc)
cv_mean = np.mean(cv_acc)
cv_std = np.std(cv_acc)

print(
    f'Training accuracy of {n_repeats}x{n_splits}-fold CV:\t{train_mean:.2} +/- {train_std:.2} (Method: PCA)'
)
print(
    f'CV accuracy of {n_repeats}x{n_splits}-fold CV:\t\t{cv_mean:.2} +/- {cv_std:.2} (Method: PCA)'
)

# %%
# Define the number of PCA components
n_pca_feats = 8

# Initialize the PCA
pca = PCA(n_pca_feats)

# Define the pipeline
pipe = Pipeline([('Standard Scaler', ss), ('PCA', pca), ('LDA', lda)])
pipe = Pipeline([('Standard Scaler', ss), ('PCA', pca), ('SVC', svc)])

# Initialize a list for the accuracies
train_acc = []
cv_acc = []

# Fit the pipeline to whole of the training data
pipe.fit(X_train, y_train)

y_pred_train = pipe.predict(X_train)
y_pred_test = pipe.predict(X_test)

# Compute the classification accuracies
train_acc = pipe.score(X_train, y_train)
test_acc = pipe.score(X_test, y_test)

print(f'Training accuracy: {train_acc:.2} (Method: PCA)')
print(f'Training accuracy: {test_acc:.2} (Method: PCA)')

# Compute the confusion matrices
cm_train = cm(y_train, y_pred_train)
cm_test = cm(y_test, y_pred_test)

plot_conf_mat(cm_train, cm_test, labels=['Left', 'Right'])

# %%
# Define the name of the pickle file
file = os.path.join('pickles', 'clf_pca.pkl')

# Open a file to dump the data
with open(file, 'wb') as pkl_file:
    # Dump the list to the pickle file
    pickle.dump(pipe, pkl_file)
# %%
