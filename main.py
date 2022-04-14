###### IMPORTS ###################
import threading
import time
import sounddevice as sd
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from scipy.io.wavfile import write
import pyttsx3  # not for final

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
save_path = "audio_data/"
model = load_model("./saved_model/WWD.h5")

##### LISTENING #########


def listener():
    print("listening")
    while True:
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()
        mfcc = librosa.feature.mfcc(y=myrecording.ravel(), sr=fs, n_mfcc=40)
        mfcc_processed = np.mean(mfcc.T, axis=0)
        # prediction_thread(mfcc_processed)
        prediction(mfcc_processed, myrecording)
        time.sleep(0.001)


##### PREDICTION  #############


def prediction(y, myrecording):
    print("predict")
    prediction = model.predict(np.expand_dims(y, axis=0))
    # if prediction[:, 1] > 0.96:
    if prediction[:, 1] > 0.99:
        print(f"Wake Word Detected)")
        print("Confidence:", prediction[:, 1])
        speak("Hello There?")
        # write it if was sucessfully
        write(save_path + "_no_verified_" +
              str(time.time()) + ".wav", fs, myrecording)

    else:
        print(f"Wake Word NOT Detected")
        print("Confidence:", prediction[:, 0])

    time.sleep(0.1)


if __name__ == "__main__":
    listener()
