from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Load data
mnist = fetch_openml('mnist_784')

# Looks Data Array - Understand the data
X, y = mnist['data'], mnist['target']
print(X.shape)
print(y.shape)

# Print Any Image
some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
           interpolation='nearest')
plt.axis('off')
plt.show()

# Check the labeled information for that picture
print(y[36000])

# Plot train data and test data
# 60000 Images for Training
# 10000 Images for Test
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Shuffle Training data
# Permutation based on number of Images
shuffle_index = np.random.permutation(60000)

X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# Try to predict some_digit = 9 using SGD (Stochastic Gradient Descent)
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)

y_train_9 = (y_train == 9)
y_test_9 = (y_test == 9)

sgd_clf = SGDClassifier(random_state=42)
forest_clf = RandomForestClassifier(random_state=42)

# Try to predict some digit with SGD Classifier

sgd_clf.fit(X_train, y_train)
print(sgd_clf.predict([some_digit]))

# Calculate scores for decision function

some_digit_scores = sgd_clf.decision_function([some_digit])
print(some_digit_scores)

# Find highest score

print(np.argmax(some_digit_scores))

# OVO strategy based on SGD Classifier

ovo_clf = OneVsOneClassifier(sgd_clf)
ovo_clf.fit(X_train, y_train)
print(ovo_clf.predict([some_digit]))
print(len(ovo_clf.estimators_))

# Training Random forest

forest_clf.fit(X_train, y_train)
print(forest_clf.predict([some_digit]))

# Predict probability

print(forest_clf.predict_proba([some_digit]))

# Calculate Cross Validation Score

print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy'))

# Scaling input to improve accuracy

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
print(cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring='accuracy'))


################################NEW: START FROM HERE##########################################

# Prepare Confusion Matrix

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
print(conf_mx)

# Plot confusion matrix as an image

plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

