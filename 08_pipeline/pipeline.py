import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


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


# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)


# 3. Create a pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(random_state=42)),
])


# 4. Train the entire pipeline
pipeline.fit(X_train, y_train)


# 5. Make predictions
y_pred = pipeline.predict(X_test)


# 6. Evaluate the pipeline
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

print("\nConfusion Matrix:")
print(cm)


# 7. Show pipeline steps
print("\nPipeline:")
print(pipeline)


# 8. Visualize the test predictions
plt.figure(figsize=(8, 6))

plt.scatter(
    X_test[y_test == 0, 0],
    X_test[y_test == 0, 1],
    color="red",
    label="Actual Fail",
    s=80,
)

plt.scatter(
    X_test[y_test == 1, 0],
    X_test[y_test == 1, 1],
    color="blue",
    label="Actual Pass",
    s=80,
)

plt.xlabel("Study Hours")
plt.ylabel("Practice Tests")
plt.title("Pipeline Test Data")
plt.legend()
plt.grid(True)

plt.show()