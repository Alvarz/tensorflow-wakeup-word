###### IMPORTS ###################
import threading
import time
import sys
import sounddevice as sd
import librosa
import numpy as np
from tensorflow.keras.models import load_model

from scipy.io.wavfile import write
import pyttsx3

#### SETTING UP TEXT TO SPEECH ###
engine = pyttsx3.init()


def speak(audio):
    engine.say(audio)
    engine.runAndWait()
    if engine._inLoop:
        engine.endLoop()


##### CONSTANTS ################
fs = 22050
seconds = 2
save_path = "unverified/"
model = load_model("./saved_model/WWD.h5")

##### LISTENING THREAD #########

listen_thread = None
pred_thread = None


def listener():
    while True:
        try:
            myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
            sd.wait()
            mfcc = librosa.feature.mfcc(
                y=myrecording.ravel(), sr=fs, n_mfcc=40)
            mfcc_processed = np.mean(mfcc.T, axis=0)
            prediction_thread(mfcc_processed, myrecording)
            time.sleep(0.001)
        except KeyboardInterrupt:
            print("Stopping treads")
            listen_thread.stop = True
            pred_thread.stop = True
            sys.exit()


def voice_thread():
    listen_thread = threading.Thread(target=listener, name="ListeningFunction")
    listen_thread.start()

##### PREDICTION THREAD #############


def prediction(y, myrecording):
    prediction = model.predict(np.expand_dims(y, axis=0))
    if prediction[:, 1] > 0.99:
        print(f"Wake Word Detected)")
        print("Confidence:", prediction[:, 1])
        speak("Hello There?")
        # write it if was sucessfully
        write(save_path +
              str(time.time()) + ".wav", fs, myrecording)

    else:
        print(f"Wake Word NOT Detected")
        print("Confidence:", prediction[:, 0])

    time.sleep(0.1)


def prediction_thread(y, myrecording):
    pred_thread = threading.Thread(
        target=prediction, name="PredictFunction", args=(y, myrecording,))
    pred_thread.start()


voice_thread()
