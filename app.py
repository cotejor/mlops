#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')

import numpy as np

from flask import Flask, request, json, jsonify

import json
import os
import pickle

import predictions as pr

app = Flask(__name__)

@app.route("/")
def hello():
    return jsonify(message="Hello! I'm another Iris Classifier")


@app.route('/classify', methods=['GET'])
def classify():
    """
        Classify the flowers based on sepal and petal measures.
    """
    # Get the url params.
    var1 = request.args.get('var1')
    var2 = request.args.get('var2')
    
    sample = [var1, var2]

    # Validations 101
    if None in sample:
        return jsonify(message='Missing some measure'), 404

    for measure in sample:
        try:
            float(measure)
        except ValueError:
            return jsonify(message='Some measure is not a float'), 404

    return jsonify(pr.predict_proba(sample))


if __name__ == '__main__':    
    app.run(host='0.0.0.0', debug=True) 