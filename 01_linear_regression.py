# Experiment 1: Single-variable Linear Regression (Normal Equation + Gradient Descent)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load data (first run needs internet; cached data works offline)
d = fetch_california_housing()
X = d.data[:, [d.feature_names.index('AveRooms')]]
y = d.target
mask = X[:, 0] < 15                 # remove extreme room-count outliers
X, y = X[mask], y[mask]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
s = StandardScaler()
Xtr, Xte = s.fit_transform(Xtr), s.transform(Xte)
Xtr = np.c_[np.ones(len(Xtr)), Xtr] # add intercept column
Xte = np.c_[np.ones(len(Xte)), Xte]

# Method 1: Normal Equation
theta_ne = np.linalg.pinv(Xtr.T @ Xtr) @ Xtr.T @ ytr

# Method 2: Gradient Descent
np.random.seed(42)
theta_gd = np.random.randn(2)
for _ in range(1000):
    grad = 2 / len(Xtr) * Xtr.T @ (Xtr @ theta_gd - ytr)
    theta_gd -= 0.05 * grad

for name, theta in [('Normal Equation', theta_ne), ('Gradient Descent', theta_gd)]:
    pred = Xte @ theta
    print(name, 'MSE:', mean_squared_error(yte, pred), 'R2:', r2_score(yte, pred))

order = np.argsort(Xte[:, 1])
plt.scatter(Xte[:, 1], yte, s=10, label='Actual')
plt.plot(Xte[order, 1], (Xte @ theta_gd)[order], 'r', label='Gradient Descent')
plt.plot(Xte[order, 1], (Xte @ theta_ne)[order], 'g--', label='Normal Equation')
plt.xlabel('Standardized AveRooms'); plt.ylabel('House value'); plt.legend(); plt.show()
