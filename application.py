# app.py - Simple Flask App for Operation Sound Sentinel
# Run with: flask run
# Requires: pip install flask torch sounddevice scipy numpy python_speech_features
# Note: Assumes pre-trained 'net' model is defined/loaded. For demo, use a dummy model.
#       sounddevice needs microphone access; test on local machine.

import os
import threading
import time
from datetime import datetime
from flask import Flask, render_template, jsonify
import numpy as np
import torch
import sounddevice as sd
from scipy.io import wavfile
from python_speech_features import mfcc
from src.utils.model import AudioCNN
#os.chdir("../")

app = Flask(__name__)

# Config
SAMPLE_RATE = 48000
DURATION = 4
THRESHOLD = 0.5
device = torch.device('cpu')
counts = 0
logs = []  # Simple in-memory log for detections



net = AudioCNN(n_classes=1).to(device)  # Load your real model here, e.g., net.load_state_dict(torch.load('model.pth'))
net.load_state_dict(torch.load("final_model/model.pt"))

def extract_features(audio, rate=SAMPLE_RATE):
    X = []
    _min, _max = float("inf"), -float("inf")
    try:
        os.makedirs('data_from_user/userinput', exist_ok=True)
        filename = f"data_from_user/userinput/{counts}.wav"
        wavfile.write(filename=filename, rate=rate, data=audio)
        
        wav = audio[:rate * DURATION]
        
        X_sample = mfcc(
            wav, rate,
            numcep=13,
            nfilt=26,
            nfft=512
        ).T
        
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample)
        
    except Exception as e:
        print(f"Error: {e}")
    
    if not X:
        return torch.tensor([]).float()
    
    X = np.array(X)
    if _max > _min:
        X = (X - _min) / (_max - _min)
    X = X[..., np.newaxis]
    X = torch.tensor(X).float().permute(0, 3, 1, 2).to(device)
    return X

def predict(audio):
    features = extract_features(audio)
    if features.numel() == 0:
        return 0.0
    with torch.no_grad():
        output = net(features)
        prob = torch.sigmoid(output).item()
    return prob

def background_listener():
    global counts
    print("🎧 Starting background listener...")
    while True:
        try:
            audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype=np.float32)
            sd.wait()
            audio = audio.flatten()
            
            prob = predict(audio)
            counts += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            detection = "Gunshot" if prob >= THRESHOLD else "Background"
            emoji = "🚨" if prob >= THRESHOLD else "🔈"
            log_entry = {
                'timestamp': timestamp,
                'detection': detection,
                'probability': round(prob, 3),
                'emoji': emoji
            }
            logs.append(log_entry)
            if len(logs) > 10:  # Keep last 10
                logs.pop(0)
            
            # Simulate notification (print for now; add email/SMS here)
            if prob >= THRESHOLD:
                print(f"{emoji} {detection} detected! Prob: {prob:.3f} at {timestamp}")
                # e.g., send_sms(f"Alert: {detection} at {timestamp}")
            
            time.sleep(1)  # Short pause
        except Exception as e:
            print(f"Listener error: {e}")
            time.sleep(5)

# Start listener in background thread
listener_thread = threading.Thread(target=background_listener, daemon=True)
listener_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/logs')
def get_logs():
    return jsonify({'logs': logs[::-1]})  # Recent first

if __name__ == '__main__':
    os.makedirs('data_from_user/userinput', exist_ok=True)
    app.run(debug=True,host="0.0.0.0", port=5000)
