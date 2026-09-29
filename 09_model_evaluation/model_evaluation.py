import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


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


# 3. Train a model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42,
)

model.fit(X_train, y_train)


# 4. Make predictions
y_pred = model.predict(X_test)


# 5. Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)


# 6. Print results
print("Model Evaluation")
print("-----------------")

print(f"Accuracy : {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall   : {recall:.2f}")
print(f"F1 Score : {f1:.2f}")

print("\nConfusion Matrix:")
print(cm)


# 7. Visualize the confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fail", "Pass"],
)

disp.plot()
plt.title("Confusion Matrix")
plt.show()