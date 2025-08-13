Code :

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
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

num_samples = 4
fig, axs = plt.subplots(num_samples, 2, figsize=(6, num_samples * 2))
fig.suptitle("Input Image    True Label    Predicted Label    Correct (Y/N)", fontsize=12, fontweight='bold')

for i in range(num_samples):
    axs[i, 0].imshow(x_test[i], cmap='gray')
    axs[i, 0].axis('off')
    true_lbl = class_names[y_test[i]]
    pred_lbl = class_names[y_pred[i]]
    correct = 'Y' if y_pred[i] == y_test[i] else 'N'
    text = f"{true_lbl:<12} {pred_lbl:<16} {correct}"
    axs[i, 1].axis('off')
    axs[i, 1].text(0, 0.5, text, fontsize=10, va='center', fontfamily='monospace')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

Output :
<img width="518" height="635" alt="image" src="https://github.com/user-attachments/assets/25e10ade-7346-432c-acc1-c1408351b4aa" />

