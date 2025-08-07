Code :

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0], [1], [1], [0]])

model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, Y, epochs=1000, verbose=0)
loss, acc = model.evaluate(X, Y)
print("Accuracy:", acc)
print("Predictions:", model.predict(X))

plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

Output :

<img width="1822" height="218" alt="Screenshot 2025-08-07 184253" src="https://github.com/user-attachments/assets/96bf0014-2214-4895-a2d8-3f2d4489ce77" />
<img width="1075" height="568" alt="Screenshot 2025-08-07 184303" src="https://github.com/user-attachments/assets/ac0305d1-c120-447c-bb79-509b215f2c40" />

