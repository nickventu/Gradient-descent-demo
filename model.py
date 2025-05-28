import numpy as np
import matplotlib.pyplot as plt


x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
y = np.array([3, 5, 7, 9, 11], dtype=np.float32)

m = 0
b = 0
alpha = 0.01
epochs = 1000
n = len(x)

for _ in range(epochs):
    y_pred = m * x + b
    error = y - y_pred

    #Computing gradients
    dm = (-2/n) * sum(error * x)
    db = (-2/n) * sum(error)

    #Update parameters
    m = m - alpha * dm
    b = b - alpha * db

y_pred = [m * xi + b for xi in x]

# Plot the original data
plt.scatter(x, y, color='blue', label='Actual data')

# Plot the predicted line
plt.plot(x, y_pred, color='red', label='Model prediction')

# Labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.legend()
plt.grid(True)
plt.show()

print(f"Learned model: y = {m:.2f}x + {b:.2f}")
