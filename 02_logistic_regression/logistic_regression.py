import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


# 1. Create sample data
# X = study hours
# y = 0: fail, 1: pass

X = np.array([
    [1],
    [2],
    [2.5],
    [3],
    [3.5],
    [4],
    [4.5],
    [5],
    [5.5],
    [6],
    [6.5],
    [7],
    [7.5],
    [8],
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
])


# 2. Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# 3. Create the model
model = LogisticRegression()


# 4. Train the model
model.fit(X_train, y_train)


# 5. Make predictions
y_pred = model.predict(X_test)


# 6. Get prediction probabilities
y_prob = model.predict_proba(X_test)[:, 1]


# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


print("Coefficient:", model.coef_[0][0])
print("Intercept:", model.intercept_[0])
print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(cm)

print("\nPredictions:")
for hours, actual, predicted, probability in zip(
    X_test.flatten(),
    y_test,
    y_pred,
    y_prob,
):
    print(
        f"Study hours: {hours:.1f}, "
        f"Actual: {actual}, "
        f"Predicted: {predicted}, "
        f"Pass probability: {probability:.2f}"
    )


# 8. Visualize the data and probability curve
X_curve = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_curve = model.predict_proba(X_curve)[:, 1]

plt.scatter(X, y, label="Actual data")
plt.plot(X_curve, y_curve, label="Pass probability")

plt.axhline(0.5, linestyle="--", label="Decision threshold")

plt.xlabel("Study hours")
plt.ylabel("Probability of passing")
plt.title("Logistic Regression")
plt.legend()
plt.show()