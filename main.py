import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from random import randint

data = pd.read_csv('iris.data', header=None)
features = data.iloc[:, :4].to_numpy()
labels = data.iloc[:, 4].to_numpy()

encoder = preprocessing.LabelEncoder()
encoded_labels = encoder.fit_transform(labels)
X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.5)

# 1. calculate number of misclassified observations
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
misclassified = np.count_nonzero(y_test != y_pred)
print("Number of misclassified observations:", misclassified)

# 2. calculate classification accuracy
accuracy = gnb.fit(X_train, y_train).score(X_test, y_test) * 100
print("Classification accuracy:", accuracy)

# 3. plot classification accuracy and percentage of misclassified observations against test set size
test_sizes = []
misclassification_percentages = []
accuracies = []

size = 0
while size <= 0.95:
    size += 0.05
    X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=size)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    test_sizes.append(size)
    misclassification_percentages.append(np.count_nonzero(y_test != y_pred) / len(y_pred))
    accuracies.append(gnb.fit(X_train, y_train).score(X_test, y_test))

# create bar plot
fig, ax = plt.subplots()
ax.bar(test_sizes, accuracies, width=0.03, color='dodgerblue', label='Accuracy')
ax.bar(test_sizes, misclassification_percentages, width=0.03, color='salmon', label='% Misclassified')
ax.set_facecolor('seashell')
fig.set_figwidth(17)
fig.set_figheight(8)
fig.set_facecolor('floralwhite')
ax.legend()
ax.set_xlabel('Test Set Size')
ax.set_ylabel('Accuracy/Misclassification %')
ax.set_title('Classification Accuracy and Misclassification % vs Test Set Size')
plt.show()

# 1. calculate number of misclassified observations for MultinomialNB
clf = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
y_pred = clf.fit(X_train, y_train).predict(X_test)
misclassified = (y_test != y_pred).sum()
print("Number of misclassified observations for MultinomialNB:", misclassified)

print(np.count_nonzero(y_test != y_train))
print(clf.fit(X_train, y_train).score(X_test, y_test) * 100)
clf = ComplementNB(force_alpha=True)
Y_pred = clf.fit(X_train, y_train).predict(X_test)
print(f'Number of observations that were misclassified: {np.count_nonzero(y_test != Y_pred)}')
print(f'Classification accuracy: {clf.fit(X_train, y_train).score(X_test, y_test) * 100}%')
clf = BernoulliNB(force_alpha=True)
Y_pred = clf.fit(X_train, y_train).predict(X_test)
print(f'Number of observations that were misclassified: {np.count_nonzero(y_test != Y_pred)}')
print(f'Classification accuracy: {clf.fit(X_train, y_train).score(X_test, y_test) * 100}%')
clf = tree.DecisionTreeClassifier()
Y_pred = clf.fit(X_train, y_train).predict(X_test)
print(np.count_nonzero(y_test != Y_pred))
print(f'Classification accuracy: {clf.fit(X_train, y_train).score(X_test, y_test) * 100}%')
#3
print(f'Number of leaves: {clf.get_n_leaves()}')
print(f'Depth: {clf.get_depth()}')
#4
plt.subplots(1, 1, figsize=(10, 10))
tree.plot_tree(clf, filled=True)
plt.show()
#5
size = 0
list_test_size = []
percentage_misclassified_observations = []
classification_accuracy = []
while size <= 0.95:
    size += 0.05
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=size)
    gnb = tree.DecisionTreeClassifier()
    Y_pred = gnb.fit(X_train, Y_train).predict(X_test)
    list_test_size.append(size)
    percentage_misclassified_observations.append(np.count_nonzero(Y_test != Y_pred) / len(Y_pred))
    classification_accuracy.append(gnb.fit(X_train, Y_train).score(X_test, Y_test))
fig, ax = plt.subplots()
ax.bar(list_test_size, classification_accuracy, width=0.03, color='mediumaquamarine')
ax.bar(list_test_size, percentage_misclassified_observations, width=0.03, color='coral')
ax.set_facecolor('lightsteelblue')
fig.set_figwidth(17)
fig.set_figheight(8)
fig.set_facecolor('lightgray')
plt.xlabel('Test size')
plt.ylabel('Accuracy and percentage of misclassified observations')
plt.title('Accuracy and percentage of misclassified observations vs. test size')
plt.show()
#6
criterion_parameters = ('gini', 'entropy', 'log_loss')
splitter_parameter = ('best', 'random')
for parameter in criterion_parameters:
    sp_par_random = splitter_parameter[randint(0, 1)]
    max_dp_random = randint(5, 40)
    min_samples_split_random = randint(5, 40)
    min_samples_leaf_random = randint(5, 40)
    gnb = tree.DecisionTreeClassifier(criterion=parameter, splitter=sp_par_random, max_depth=max_dp_random,
                                      min_samples_split=min_samples_split_random,
                                      min_samples_leaf=min_samples_leaf_random)
    Y_pred = gnb.fit(X_train, Y_train).predict(X_test)
    print(f'With criterion: {parameter}, splitter: {sp_par_random}, max_depth: {max_dp_random}, min_samples_split: {min_samples_split_random}, min_samples_leaf: {min_samples_leaf_random} \n classification accuracy: {gnb.fit(X_train, Y_train).score(X_test, Y_test) * 100}%, number of leaves: {gnb.get_n_leaves()}, depth: {gnb.get_depth()}\n')