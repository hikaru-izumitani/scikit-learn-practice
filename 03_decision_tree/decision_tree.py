import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


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


# 2. Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)


# 3. Create the model
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42,
)


# 4. Train the model
model.fit(X_train, y_train)


# 5. Make predictions
y_pred = model.predict(X_test)


# 6. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(cm)

print("\nPredictions:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual}, Predicted: {predicted}")


# 7. Visualize the decision tree
plt.figure(figsize=(12, 8))

plot_tree(
    model,
    feature_names=["Study Hours", "Practice Tests"],
    class_names=["Fail", "Pass"],
    filled=True,
)

plt.title("Decision Tree")
plt.show()