Code :

import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, verbose=0)

y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

indices = [0, 1, 2, 3]

print(f"{'Input Digit Image':<18} {'Expected Label':<15} {'Model Output':<13} {'Correct (Y/N)'}")
for i in indices:
    expected = y_test[i]
    predicted = y_pred[i]
    correct = 'Y' if expected == predicted else 'N'
    print(f"Image of {expected:<9} {expected:<15} {predicted:<13} {correct}")

Output :

<img width="1536" height="153" alt="image" src="https://github.com/user-attachments/assets/5862fe74-7bfb-478c-b180-2791420bb448" />

