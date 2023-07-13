import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import confusion_matrix
iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.75, random_state=42)


def knn(X_train, y_train, X_test, k=5):
    y_pred = []
    for x in X_test:
        distances = np.linalg.norm(x-X_train, axis=1)
        indices = np.argsort(distances)[:k]
        labels = y_train[indices]
        label = Counter(labels).most_common(1)[0][0]
        y_pred.append(label)
    return y_pred


y_pred = (knn(X_train, y_train, X_test, 5))
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy*100, "%")


CM = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(CM)
