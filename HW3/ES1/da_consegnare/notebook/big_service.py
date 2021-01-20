import cherrypy
import json
import base64
import numpy as np
import tensorflow as tf

#preprocess audio bytes to mfcc
sampling_rate = 16000
frame_length = 640
frame_step = 320
lower_frequency = 20
upper_frequency = 4000
num_mel_bins = 40
num_coefficients = 10
num_spectrogram_bins = (frame_length) // 2 + 1

linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sampling_rate,
                                                                    lower_frequency, upper_frequency)

def pad(audio):
    zero_padding = tf.zeros([sampling_rate] - tf.shape(audio), dtype=tf.float32)
    audio = tf.concat([audio, zero_padding], 0)
    audio.set_shape([sampling_rate])
    return audio

def get_spectrogram(audio):
    stft = tf.signal.stft(audio, frame_length=frame_length,
    frame_step=frame_step, fft_length=frame_length)
    spectrogram = tf.abs(stft)
    return spectrogram

def get_mfccs(spectrogram):
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs = mfccs[..., :num_coefficients]
    return mfccs

def process_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=1)
    audio = pad(audio)
    spectrogram = get_spectrogram(audio)
    mfccs = get_mfccs(spectrogram)
    mfccs = tf.expand_dims(mfccs, -1)
    return mfccs

#load tflite model from memory
# Load TFLite model and allocate tensors.
#the model must be in the same folder as the script
interpreter = tf.lite.Interpreter(model_path="big_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


input_shape = input_details[0]['shape']

class myService(object):
    exposed = True

    def GET(self, *path, **query):

        return("ciao")

    def POST(self, *path, **query):
        # receives a json containing a raw audio file
        # reads the body of the http request
        # processes the audio file as mfcc
        # runs inference with big model using mfcc as input
        # returns the values of the last layer of the neural network

        ##simplified version

        # read the http body and return a json string
        req = cherrypy.request.body.read()
        # convert the json string into python dictionary
        body = json.loads(req)
        audio_string = body["e"][0]["vd"]
        audio_binary = base64.b64decode(audio_string)
        mfccs = process_audio(audio_binary)
        mfccs = np.expand_dims(mfccs, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], mfccs)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        response = {
            'pred' : int(np.argmax(output_data))
        }
        response = json.dumps(response)

        return response

    def PUT(self, *path, **query):
        pass

    def DELETE(self, *path, **query):
        pass

if __name__ == '__main__':
    conf = {
        '/': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
            'tools.sessions.on': True,
        }
    }
    cherrypy.tree.mount (myService(), "/",conf)
    cherrypy.config.update({'server.socket_host': '192.168.178.107'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()
