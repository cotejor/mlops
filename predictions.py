#!/usr/bin/env python3
import numpy as np

import os
import pickle
import random

from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.models import model_from_json

from config import *
import aws_utils as au

""" Fetch trained models, encoders and tokenizers. Make predictions. """
def fetch_pickle(file_name: str):
       
    with open(os.path.join(folder, file_name), 'rb') as f:
        fetched_object = pickle.load(f)

    return fetched_object    

def predict(sample):
    return fetch_pickle("./dist/deepgmi.pkl").predict(sample)

