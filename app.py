from flask import Flask, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# load models and data
nn_model = load_model('best_nn_model.keras')
rf_model = joblib.load('rf_model.pkl')
sgd_model = joblib.load('sgd_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')
model_info = joblib.load('model_info.pkl')
test_set = pd.read_csv('test_set.csv')

def preprocess_song_data(song_data):
  # create dataframe
  df = pd.DataFrame([song_data])
  
  # one-hot encode
  if 'key' in df.columns:
    df = pd.get_dummies(df,  columns=['key'],  drop_first=True)
  if 'mode' in df.columns:
    df = pd.get_dummies(df,  columns=['mode'],  drop_first=True)
  

  # add missing features
  for feature in model_info['feature_names']:
    if feature not in df.columns:
      df[feature] =0
  
  # reorder and scale
  df = df.reindex(columns=model_info['feature_names'], fill_value=0)
  return scaler.transform(df)

@app.route("/")
def index():
  return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
  
  # Sample from test set (unseen data)
  random_song = test_set.sample(1).iloc[0]
  song_features = random_song.drop(['music_genre'], errors='ignore').to_dict()
  
  # clean missing values (replace '?' with 0)
  for key, value in song_features.items():
    if value == '?' or value == '' or pd.isna(value):
      song_features[key] = 0
  
  song_info = {
    'actual_genre': random_song.get('music_genre', 'Unknown')
  }

  # predict
  processed_data = preprocess_song_data(song_features)
  
  nn_prediction =nn_model.predict(processed_data,verbose=0)
  rf_prediction =rf_model.predict_proba(processed_data)
  sgd_prediction = sgd_model.predict_proba(processed_data)
  
  # get results
  nn_index=  np.argmax(nn_prediction[0])
  rf_index= np.argmax(rf_prediction[0])
  sgd_index= np.argmax(sgd_prediction[0])
  
  all_probs = {}
  for i, genre in enumerate(label_encoder.classes_):
    all_probs[genre] = {
      'nn_prob':  float(nn_prediction[0][i]),
      'rf_prob': float(rf_prediction[0][i]),
      'sgd_prob': float(sgd_prediction[0][i])
    }
  
  return jsonify({
    'song_info': song_info,
    'predictions': {
      'neural_network': {
        'genre': label_encoder.classes_[nn_index],
        'confidence': float(nn_prediction[0][nn_index])
      },
      'random_forest': {
        'genre': label_encoder.classes_[rf_index],
        'confidence': float(rf_prediction[0][rf_index])
      },
      'sgd_classifier': {
        'genre': label_encoder.classes_[sgd_index],
        'confidence': float(sgd_prediction[0][sgd_index])
      }
    },
    'all_probabilities': all_probs
    })

if __name__ == "__main__":
  print("http://localhost:8080")
  app.run(host='0.0.0.0', port=8080) 