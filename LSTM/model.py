import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import plotly.express as px
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
import re
from tensorflow.keras.layers import Embedding, Dense, Dropout, LSTM, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import utils

from sklearn.metrics import confusion_matrix, classification_report

import warnings
warnings.filterwarnings("ignore")

# =====================
# CONFIG
# =====================
train_path = 'DataSet/train.ft.txt.bz2/train.ft.txt.bz2'
test_path = 'DataSet/test.ft.txt.bz2/test.ft.txt.bz2'

MODEL_PATH = "saved_model/lstm_model.h5"

EPOCHS = 10
BATCH_SIZE = 64

# =====================
# LOAD DATA
# =====================
print("Loading dataset...")
train_data=pd.read_csv(train_path,compression='bz2',delimiter='\t')
test_data=pd.read_csv(test_path,compression='bz2',delimiter='\t')

'''Build a function to convert data into a data frame with 2 columns: label, text,
The function takes the file or the train/test data
and a for loop loops over the text in the file to split texts from labels'''

def process_data(file):
    data = []
    for index, row in file.iterrows():
         # first line data is raw data
        line = row[0]

        #split lines into text and labels
        label, text = line.split(' ', 1)

        #remove the __label__ only keep the number
        label = label.replace('__label__', '')

        #append
        data.append((label, text.strip()))

    cols = ['label', 'review']
    return pd.DataFrame(data, columns=cols)


def text_cleaning(text):
  #convert to lower case
  text=text.lower()

  #remove special characters and numsbers and extra whitespace
  pattern_punc = r'[^a-zA-Z\s]'
  text = re.sub(pattern_punc, '', text)
  return text



train=process_data(train_data)
test=process_data(test_data)

train['label'] = train['label'].replace({"2": 1, "1": 0}).astype(int)
test['label']  = test['label'].replace({"2": 1, "1": 0}).astype(int)


train['review_cleaned']=train['review'].apply(text_cleaning)
test['review_cleaned']=test['review'].apply(text_cleaning)

# =====================
# TOKENIZATION AND PADDING
# =====================
max_words = 1000
max_len = 100

tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(train['review_cleaned'])

X_train = tokenizer.texts_to_sequences(train['review_cleaned'])
X_test = tokenizer.texts_to_sequences(test['review_cleaned'])

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

y_train=train['label']
y_test=test['label']

# =====================
# TRAIN / TEST SPLIT
# =====================
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# =====================
# MODEL
# =====================



model = Sequential()
model.add(Input(shape=(max_len,), dtype='int32')) 
model.add(Embedding(input_dim=max_words, output_dim=128))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


model.summary()




# =====================
# TRAIN
# =====================




print("Training started...")
model.fit(X_train,y_train,validation_data=(X_valid,y_valid),epochs=5,batch_size=64, verbose=1)

# =====================
# SAVE MODEL
# =====================

model.save('sentiment_lstm_model.h5')
print("Training completed. Model saved.")