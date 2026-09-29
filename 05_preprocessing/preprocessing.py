import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


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


# 3. Create the scaler
scaler = StandardScaler()


# 4. Fit the scaler using training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using the same scaler
X_test_scaled = scaler.transform(X_test)


# 5. Create and train the model
model = LogisticRegression(random_state=42)

model.fit(X_train_scaled, y_train)


# 6. Make predictions
y_pred = model.predict(X_test_scaled)


# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")


# 8. Compare original and scaled data
print("\nOriginal training data:")
print(X_train)

print("\nScaled training data:")
print(X_train_scaled)


# 9. Visualize the scaled data
plt.figure(figsize=(8, 6))

plt.scatter(
    X_train_scaled[y_train == 0, 0],
    X_train_scaled[y_train == 0, 1],
    color="red",
    label="Fail",
    s=60,
)

plt.scatter(
    X_train_scaled[y_train == 1, 0],
    X_train_scaled[y_train == 1, 1],
    color="blue",
    label="Pass",
    s=60,
)

plt.xlabel("Study Hours (Standardized)")
plt.ylabel("Practice Tests (Standardized)")
plt.title("Standardized Training Data")
plt.legend()
plt.grid(True)

plt.show()