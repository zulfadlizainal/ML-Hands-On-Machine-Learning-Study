from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score

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
sgd_clf.fit(X_train, y_train_9)

# Now can use the SGD to detect the image of number 9
print(sgd_clf.predict([some_digit]))

# Predict using SGD & check Condusion matrix

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_9, cv=3)
print(confusion_matrix(y_train_9, y_train_pred))

# Calculate Precision
print(precision_score(y_train_9, y_train_pred))

# Calculate Recall
print(recall_score(y_train_9, y_train_pred))

################################NEW: START FROM HERE##########################################

# Calculate F1 Score
print(f1_score(y_train_9, y_train_pred))
