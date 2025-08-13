Code :

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_acc:.4f}")

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
])
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()

comparison_data = {
    "Dataset": ["MNIST Digits", "Fashion-MNIST"],
    "Model Accuracy": ["> 97%", "89–91%"],
    "Notes": ["Simpler features, easier task", "Higher difficulty, real-world"]
}
comparison_df = pd.DataFrame(comparison_data)
print("\nComparison Table:\n")
print(comparison_df.to_string(index=False))

Output:

<img width="1535" height="277" alt="image" src="https://github.com/user-attachments/assets/6f8ba071-27fe-43e3-a769-1bd93363acbd" />
<img width="776" height="490" alt="image" src="https://github.com/user-attachments/assets/58ea2276-5510-4006-ad45-2e63b8605a54" />
<img width="493" height="101" alt="image" src="https://github.com/user-attachments/assets/878a4ae0-76f7-46bc-8ff6-786cad51a60a" />

