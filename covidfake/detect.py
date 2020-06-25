import os
import tensorflow as tf 
import numpy as np

import keras
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

porter_stemmer = PorterStemmer()
stop = stopwords.words('english')

PATH = os.getcwd()
model = load_model(f"{PATH}/covidfake/models/ConvolutionalModel.h5")  
model.summary()

embeddings_index = {}
with open('/tmp/glove.6B.100d.txt') as f:
  for line in f:
    values = line.split();
    word = values[0];
    coefs = np.asarray(values[1:], dtype='float32');
    embeddings_index[word] = coefs;

def __encode(text):
  tokens = word_tokenize(text)
  stems = [porter_stemmer.stem(t) for t in tokens if not t in stop]
  stemmed_str = ' '.join(stems)
  
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts([stemmed_str])
  
  word_index = tokenizer.word_index
  vocab_size = len(word_index.keys())

  sequences = tokenizer.texts_to_sequences([stemmed_str])
  padded = pad_sequences(sequences, maxlen=500, padding='post', truncating='post')
    
  return model.predict([padded])[0][0]

def detect_fake(text):
  result = __encode(text)
  return f"Stemmization sentence: {result}"

# Comments:
# - stopwords are applied AFTER the stemmization, which might lead to inaccurate results.