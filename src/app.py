from xgboost import XGBClassifier
import flask
import config
import time
import pickle
import numpy as np
from flask import Flask, request
from utils import extract_features

app = Flask(__name__)

MODEL = None


def audio_prediction(audio_file):
    feats = extract_features(
        audio_file, mel=True, mfcc=True, chroma=True, contrast=True)
    scaler = pickle.load(open(config.SCALAR_PATH, 'rb'))
    X = scaler.transform(feats.reshape(1, -1))
    pred = MODEL.predict_proba(X)
    return pred[0][1]


@app.route("/predict")
def predict():
    audio_file = request.args.get("audio_file")
    start_time = time.time()
    male_prediction = audio_prediction(audio_file)
    female_prediction = 1 - male_prediction
    response = {}
    response["response"] = {
        "male": str(male_prediction),
        "female": str(female_prediction),
        "audio_file": str(audio_file),
        "time_taken": str(time.time() - start_time),
    }
    return flask.jsonify(response)


if __name__ == "__main__":
    MODEL = XGBClassifier()
    MODEL.load_model(config.MODEL_PATH)
    app.run()
