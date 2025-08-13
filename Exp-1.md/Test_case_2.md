Code :

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0]) 

clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X, y)

predictions = clf.predict(X)

print("Test Input (X)  Actual Output by Perceptron  Expected  Remarks")
for i in range(len(X)):
    actual = predictions[i]
    expected = y[i]
    remark = "Correct" if actual == expected else "May fail"
    print(f"{X[i]}           {actual}                          {expected}         {remark}")

for i in range(len(X)):
    color = 'red' if y[i] == 0 else 'blue'
    plt.scatter(X[i][0], X[i][1], color=color, s=100, edgecolors='k', label=f"Class {y[i]}" if i < 2 else "")

x_values = np.array([0, 1])
if clf.coef_[0][1] != 0:
    y_values = -(clf.coef_[0][0] * x_values + clf.intercept_) / clf.coef_[0][1]
    plt.plot(x_values, y_values.flatten(), label='Decision Boundary', color='green')
else:
    print("Cannot plot decision boundary: vertical line")

plt.title('Perceptron Decision Boundary for XOR')
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.legend()
plt.grid(True)
plt.show()

Output:
<img width="620" height="514" alt="Screenshot 2025-08-13 101320" src="https://github.com/user-attachments/assets/a8940631-5366-432f-9254-6729620530bf" />

