
# Import necessary libraries
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report



df = pd.read_csv('emails.csv')

X = df['email'].astype(str)
y = df['label_num']


y = np.where(y == 'spam', 1, 0)
spam_count = len(df[df['label_num'] == 1])
ham_count = len(df[df['label_num'] == 0])

print(f"Spam Count: {spam_count}")
print(f"Ham Count: {ham_count}")



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

max_words = 10000
max_sequence_length = 200


tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length, padding='post', truncating='post')

model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_sequence_length))
model.add(LSTM(units=64, return_sequences=True))
model.add(LSTM(units=32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 3  # Number of training iterations
batch_size = 25
model.fit(X_train_pad, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test_pad, y_test))

# Evaluate the model
y_pred = (model.predict(X_test_pad) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the model's accuracy and classification report
print(f'Accuracy: {accuracy}')
print(report)


