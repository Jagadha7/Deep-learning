Code :
# Step 1: Install dependencies
!pip install tensorflow

# Step 2: Import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, TimeDistributed, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Step 3: Sample data
sentences = [["John", "lives", "in", "New", "York"]]
tags = [["B-PER", "O", "O", "B-LOC", "I-LOC"]]

# Step 4: Create vocab and tag mappings
words = list(set(w for s in sentences for w in s))
tags_list = list(set(t for ts in tags for t in ts))

word2idx = {w: i + 2 for i, w in enumerate(words)}
word2idx["PAD"] = 0
word2idx["UNK"] = 1

tag2idx = {t: i for i, t in enumerate(tags_list)}

# Step 5: Convert words and tags to indices
X = [[word2idx.get(w, word2idx["UNK"]) for w in s] for s in sentences]
y = [[tag2idx[t] for t in ts] for ts in tags]

# Step 6: Pad sequences
max_len = max(len(s) for s in sentences)
X = pad_sequences(X, maxlen=max_len, padding='post')
y = pad_sequences(y, maxlen=max_len, padding='post')

# Step 7: One-hot encode labels
y = [to_categorical(i, num_classes=len(tag2idx)) for i in y]

# Step 8: Build model
input = Input(shape=(max_len,))
model = Embedding(input_dim=len(word2idx), output_dim=64, input_length=max_len)(input)
model = Bidirectional(LSTM(units=64, return_sequences=True))(model)
out = TimeDistributed(Dense(len(tag2idx), activation="softmax"))(model)

model = Model(input, out)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Step 9: Train model
model.fit(X, np.array(y), batch_size=1, epochs=10)

# Step 10: Predict
i = 0
p = model.predict(np.array([X[i]]))
p = np.argmax(p, axis=-1)

print("Word → Predicted Tag")
for w, pred in zip(sentences[i], p[0]):
    print(f"{w} → {list(tag2idx.keys())[list(tag2idx.values()).index(pred)]}")

Output :

<img width="699" height="585" alt="image" src="https://github.com/user-attachments/assets/ed1d9f01-8593-4646-b238-da39ddc19b25" />
<img width="582" height="233" alt="image" src="https://github.com/user-attachments/assets/381ca37c-84ba-4efe-80d6-407092f6cdc5" />

