import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# 1. Create sample data
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10],
])

y = np.array([
    3,
    5,
    7,
    9,
    11,
    13,
    15,
    17,
    19,
    21,
])


# 2. Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# 3. Create the model
model = LinearRegression()


# 4. Train the model
model.fit(X_train, y_train)


# 5. Make predictions
y_pred = model.predict(X_test)


# 6. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# 7. Display model information
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)
print("MSE:", mse)
print("R²:", r2)


# 8. Visualize the data and regression line
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)

plt.scatter(X, y, label="Data")
plt.plot(X_line, y_line, label="Regression line")

plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression")
plt.legend()
plt.show()