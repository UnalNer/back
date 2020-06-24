import os
import keras
import tensorflow as tf 
from keras.models import load_model

PATH = os.getcwd()
print(f"{PATH}/covidfake/models/ConvolutionalModel.h5")

model = load_model(f"{PATH}/covidfake/models/ConvolutionalModel.h5")
model.summary()

def detect_fake(text):
  return "Detecting: " + str(text)