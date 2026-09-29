import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# 1. Create a simple dataset
# X = [Study Hours, Practice Tests]
X = np.array([
    [1, 0],
    [1.5, 0],
    [2, 1],
    [2.5, 1],
    [3, 1],
    [3.5, 2],
    [4, 2],
    [4.5, 2],
    [5, 3],
    [5.5, 3],
    [6, 3],
    [6.5, 4],
    [7, 4],
    [7.5, 4],
    [8, 5],
    [8.5, 5],
])

# y = 0: Fail, 1: Pass
y = np.array([
    0, 0, 0, 0,
    0, 1, 1, 1,
    1, 1, 1, 1,
    1, 1, 1, 1,
])


# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)


# 3. Create and train the Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42,
)

model.fit(X_train, y_train)


# 4. Make predictions
y_pred = model.predict(X_test)


# 5. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(cm)


# 6. Visualize the decision regions
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300),
)

grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.2,
)

plt.scatter(
    X[y == 0, 0],
    X[y == 0, 1],
    color="red",
    label="Fail",
    s=60,
)

plt.scatter(
    X[y == 1, 0],
    X[y == 1, 1],
    color="blue",
    label="Pass",
    s=60,
)

plt.xlabel("Study Hours")
plt.ylabel("Practice Tests")
plt.title("Random Forest Decision Regions")
plt.legend()
plt.grid(True)

plt.show()