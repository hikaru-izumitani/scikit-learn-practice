import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


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
    random_state=42,
)


# 3. Define hyperparameters to test
param_grid = {
    "n_estimators": [10, 50, 100],
    "max_depth": [1, 2, 3, 5],
}


# 4. Search for the best combination
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
)

grid_search.fit(X, y)


# 5. Display the best result
print("Best parameters:")
print(grid_search.best_params_)

print(f"\nBest cross-validation accuracy:")
print(f"{grid_search.best_score_:.2f}")


# 6. Display all results
print("\nAll results:")

for params, score in zip(
    grid_search.cv_results_["params"],
    grid_search.cv_results_["mean_test_score"],
):
    print(f"{params} -> {score:.2f}")


# 7. Visualize the results
results = grid_search.cv_results_

mean_scores = results["mean_test_score"]

labels = [
    f"{params['n_estimators']} trees,\ndepth={params['max_depth']}"
    for params in results["params"]
]

plt.figure(figsize=(12, 6))

plt.bar(range(len(mean_scores)), mean_scores)

plt.xlabel("Hyperparameter Combination")
plt.ylabel("Mean CV Accuracy")
plt.title("Random Forest Hyperparameter Tuning")
plt.xticks(range(len(mean_scores)), labels, rotation=45, ha="right")
plt.ylim(0, 1.1)
plt.grid(axis="y")

plt.tight_layout()
plt.show()