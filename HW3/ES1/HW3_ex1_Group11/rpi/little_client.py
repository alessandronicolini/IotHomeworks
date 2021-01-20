import requests
import os
import sys
import pathlib
import tensorflow as tf
import numpy as np
import wave
import base64
import datetime
from datetime import timezone
import json
import time

#possibly use raspberry pi ip as device name
#device_name = socket.gethostbyname(socket.gethostname())
device_name = "192.168.178.110"

# DATASET CLASS ----------------------------------------------------------------
class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step,
            num_mel_bins=None, lower_frequency=None, upper_frequency=None,
            num_coefficients=None, mfcc=False):
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_mel_bins = num_mel_bins
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.num_coefficients = num_coefficients
        num_spectrogram_bins = (frame_length) // 2 + 1

        if mfcc is True:
            self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                    self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
                    self.lower_frequency, self.upper_frequency)
            self.preprocess = self.preprocess_with_mfcc
        else:
            self.preprocess = self.preprocess_with_stft


    def read(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]
        label_id = tf.argmax(label == self.labels)
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=1)

        return audio, audio_binary, label_id


    def pad(self, audio):
        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([self.sampling_rate])

        return audio


    def get_spectrogram(self, audio):
        stft = tf.signal.stft(audio, frame_length=self.frame_length,
                frame_step=self.frame_step, fft_length=self.frame_length)
        spectrogram = tf.abs(stft)

        return spectrogram


    def get_mfccs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram,
                self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]

        return mfccs


    def preprocess_with_stft(self, file_path):
        audio, audio_binary,  label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        spectrogram = tf.image.resize(spectrogram, [32, 32])

        return spectrogram, audio_binary, label


    def preprocess_with_mfcc(self, file_path):
        audio, audio_binary, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)

        return mfccs, audio_binary, label


    def make_dataset(self, files, train):
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(self.preprocess, num_parallel_calls=4)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)
        return ds

#creazione dataset per inferenza e per trasmissione

data_dir = pathlib.Path('data/mini_speech_commands')
if not data_dir.exists():
  tf.keras.utils.get_file(
      'mini_speech_commands.zip',
      origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
      extract=True,
      cache_dir='.', cache_subdir='data')

#lista di labels
labels = open('labels.txt').readlines()[0].split()

#lista di test
test_list=[]
with open(os.path.join(sys.path[0], "kws_test_split.txt"), "r") as file:
    for line in file:
        test_list.append('.'+line[1:-1])


MFCC_OPTIONS = {'frame_length': 888, 'frame_step': 444, 'mfcc': True,'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,'num_coefficients': 10}

generator = SignalGenerator(labels, 16000, **MFCC_OPTIONS)
test_ds = generator.make_dataset(test_list, train=False)


#success checker
def success(output_data, treshold):
    data = np.squeeze(output_data, axis=0)
    data = tf.nn.softmax(data).numpy()
    sorted_indexes = np.argsort(data)

    first = data[sorted_indexes[-1]]
    #print("first label : "+str(sorted_indexes[-1]))

    second = data[sorted_indexes[-2]]
    #print("second label : "+str(sorted_indexes[-2]))

    if first - second >= treshold:
        #print(first - second)
        return True
    else:
        #print(first - second)
        return False


#load tflite model from memory
#the model must be in the same folder as the script
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="little_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']

#big little inference
weight = 0
invocations = 0
treshold = 0.4
url="http://192.168.178.107:8080"
running_corrects = 0
total_elements = 0

for test_sample, audio_binary, label in  test_ds:

    total_elements += 1
    test_sample = np.expand_dims(test_sample, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_sample)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    if success(output_data, treshold) == False:
        invocations += 1
        # prendi audio_binary e crea un json da passare al webservice
        audio_bytes = audio_binary.numpy()
        audio_b64bytes = base64.b64encode(audio_bytes)
        audio_string = audio_b64bytes.decode()
        timestamp = int(datetime.datetime.now(timezone.utc).timestamp())

        body = {
                "bn" : device_name,
                "bt" : timestamp,
                "e" :[
                        {"n":"audio", "u":"/", "t":0, "vd":audio_string}
                ]
        }

        weight += len(json.dumps(body))
        r=requests.post(url, json=body)

        if r.status_code==200:
            # ricevere la label predetta dal bigmodel
            rbody=r.json()
            prediction = rbody['pred']
        else :
            print("unsuccessful communication")

    else :
        prediction = np.argmax(output_data)

    if prediction == label :
        running_corrects += 1

accuracy = running_corrects/total_elements

print("Accuracy : %.2f %%"%(accuracy*100))
print("Communication Cost : %.2f MB"%(weight/(1024*1024)))
