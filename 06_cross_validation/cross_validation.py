import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier


# 1. Create a dataset
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


# 2. Create the model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42,
)


# 3. Perform 5-fold cross-validation
scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy",
)


# 4. Display the results
print("Cross-validation scores:")
print(scores)

print(f"\nMean accuracy: {scores.mean():.2f}")
print(f"Standard deviation: {scores.std():.2f}")


# 5. Visualize the scores
folds = np.arange(1, 6)

plt.figure(figsize=(8, 5))

plt.bar(folds, scores)

plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("5-Fold Cross-Validation")
plt.xticks(folds)
plt.ylim(0, 1.1)
plt.grid(axis="y")

plt.show()