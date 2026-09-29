import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 1. Create sample data
# X = [study hours, practice tests]
# y = 0: fail, 1: pass

X = np.array([
    [1, 0],
    [1.5, 0],
    [2, 1],
    [2.5, 0],
    [3, 1],
    [3, 2],
    [3.5, 1],
    [4, 2],
    [4.5, 2],
    [5, 1],
    [5, 3],
    [5.5, 2],
    [6, 3],
    [6.5, 3],
    [7, 4],
    [8, 4],
])

y = np.array([
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
])


# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)


# 3. Create and train the model
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42,
)

model.fit(X_train, y_train)


# 4. Make predictions
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


# 5. Create a grid for the decision boundary
x_min = X[:, 0].min() - 0.5
x_max = X[:, 0].max() + 0.5

y_min = X[:, 1].min() - 0.5
y_max = X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300),
)

grid = np.c_[xx.ravel(), yy.ravel()]

Z = model.predict(grid)
Z = Z.reshape(xx.shape)


# 6. Plot decision regions
plt.figure(figsize=(10, 7))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.2,
)


# 7. Plot actual data
plt.scatter(
    X[y == 0, 0],
    X[y == 0, 1],
    color="red",
    edgecolor="black",
    s=100,
    label="Fail",
)

plt.scatter(
    X[y == 1, 0],
    X[y == 1, 1],
    color="blue",
    edgecolor="black",
    s=100,
    label="Pass",
)


# 8. Labels
plt.xlabel("Study Hours")
plt.ylabel("Practice Tests")
plt.title("Decision Tree Classification")
plt.legend()
plt.grid(alpha=0.2)

plt.show()