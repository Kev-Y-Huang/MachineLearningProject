from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np

# Set up Flask App
app = Flask(__name__)


@app.route("/", methods=['GET'])
def classify():
    # array mapping numbers to flower names
    # classes = ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]

    # get values for each component, return error message if not a float
    try:
        values = [[float(request.args.get(component)) for component in ['Age', 'Wt', '40YD', 'Vertical', 'BenchReps', 'BroadJump', '3Cone', 'Shuttle']]]
    except TypeError:
        return "An error occured\nUsage: 127.0.0.1:5000?Age=NUM&Wt=NUM&40YD=NUM&Vertical=NUM&BenchReps=NUM&BroadJump=NUM&3Cone=NUM&Shuttle"

    # Otherwise, return the prediction.
    prediction = knn_classifier.predict(values)[0]
    return prediction


# Run the app.
app.run()

# try 127.0.0.1:5000?Age=22&Wt=199&40YD=4.58&Vertical=40&BenchReps=13&BroadJump=124&3Cone=7.22&Shuttle=4.28
