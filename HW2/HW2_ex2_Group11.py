import argparse
import os
import pathlib
import zlib
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from keras.optimizers.schedules import PolynomialDecay

import tensorflow_model_optimization as tfmot


parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True, help='model version')
args = parser.parse_args()

#***************************** CLASSES AND FUNCTION ****************************
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

        return audio, label_id

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
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        spectrogram = tf.image.resize(spectrogram, [32, 32])

        return spectrogram, label

    def preprocess_with_mfcc(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)

        return mfccs, label

    def make_dataset(self, files, train):
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(self.preprocess, num_parallel_calls=4)
        ds = ds.batch(32)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)
        return ds

# SAVE MODELS-------------------------------------------------------------------
def save_model(model, out_path):
    
    # .tflite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    model = converter.convert()

    # compressed file
    compressed_model = zlib.compress(model)
        
    # .zip format   
    with open(out_path+'.tflite.zlib', 'wb') as f:
        f.write(compressed_model)

    return os.path.getsize(out_path+'.tflite.zlib')/1024

# WEIGHTS CLUSTERING -----------------------------------------------------------
def weights_clustering(
    in_model, 
    out_path, 
    epochs=5, 
    lr=1e-3, 
    n_clusters=4, 
    only_dense=True):
    
    if only_dense:
      def apply_clustering_to_dense(layer):
        if isinstance(layer, tf.keras.layers.Dense):
          return tfmot.clustering.keras.cluster_weights(
              layer,
              number_of_clusters=n_clusters,
              cluster_centroids_init=tfmot.clustering.keras.CentroidInitialization.LINEAR
              )
        return layer

      clustered_model = tf.keras.models.clone_model(
          in_model, 
          clone_function = apply_clustering_to_dense)

    else:     
      # create model for weight clustering
      clustered_model = tfmot.clustering.keras.cluster_weights(
          in_model,
          number_of_clusters=n_clusters,
          cluster_centroids_init=tfmot.clustering.keras.CentroidInitialization.LINEAR
          )
    
    # compile model
    clustered_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['sparse_categorical_accuracy']
        )
    
    # fit model
    clustered_model.fit(
        train_ds,
        batch_size=32, 
        epochs=epochs, 
        validation_data=val_ds)
    
    # clustered model evaluation
    _, accuracy = clustered_model.evaluate(test_ds, verbose=0)
    
    # prepare for export and save
    model_for_export = tfmot.clustering.keras.strip_clustering(clustered_model)
    zip_size = save_model(model_for_export, out_path)
    
    return model_for_export, accuracy, zip_size

# POST TRAINING QUANTIZATION ---------------------------------------------------
def get_accuracy(interpreter, test_dataset):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    running_corrects = 0
    total_elements = 0

    for (batch, labels) in test_dataset:
        total_elements += len(batch)
        for test_sample, label in  zip(batch, labels):
            test_sample = np.expand_dims(test_sample, axis=0).astype(np.float32)
            interpreter.set_tensor(input_index, test_sample)
            interpreter.invoke()
            output = interpreter.get_tensor(output_index)
            pred = np.argmax(output)
            if pred == label:
              running_corrects += 1
  
    return running_corrects/total_elements

def representative_dataset():
  train_set = train_ds.unbatch().take(200)
  img_list = []
  for image in train_set:
    img_list.append(image[0].numpy())
  img_arr = np.array(img_list)

  for data in tf.data.Dataset.from_tensor_slices(img_arr).batch(1).take(100):
    yield [data]

def pt_quantization(in_model, out_path, q_type='float16'):
    converter = tf.lite.TFLiteConverter.from_keras_model(in_model)
    if q_type == 'float16':
      converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
      converter.target_spec.supported_types = [tf.float16]
    elif q_type == 'int8int16':
      converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
      converter.representative_dataset = representative_dataset
      converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    else:
      converter.optimizations = [tf.lite.Optimize.DEFAULT]
      converter.representative_dataset = representative_dataset
      
    quant_tflite_model = converter.convert()

    # compressed file
    compressed_model = zlib.compress(quant_tflite_model)

    # generate uncompressed version for latency test
    with open(out_path+'.tflite', 'wb') as f:
      f.write(quant_tflite_model)

    # generate compressed version    
    with open(out_path+'.tflite.zlib', 'wb') as f:
        f.write(compressed_model)
    
    
    interpreter = tf.lite.Interpreter(model_content=quant_tflite_model)
    interpreter.allocate_tensors()
    
    accuracy = get_accuracy(interpreter, test_ds)
    zip_size = os.path.getsize(out_path+'.tflite.zlib')/1024
    
    return accuracy, zip_size


# MODEL FUNCTION ---------------------------------------------------------------
def get_model(conv1, conv2, conv3, conv4):
    model = models.Sequential([
      layers.Input(shape=shape),
      layers.Conv2D(filters=conv1, kernel_size=[3,3], strides=[2,1], use_bias=False),
      layers.BatchNormalization(momentum=0.1),
      layers.ReLU(),
      layers.DepthwiseConv2D(kernel_size=[3,3], strides=[1,1], use_bias=False),
      layers.Conv2D(filters=conv2, kernel_size=[1,1], strides=[1,1], use_bias=False),
      layers.BatchNormalization(momentum=0.1),
      layers.ReLU(),
      layers.DepthwiseConv2D(kernel_size=[3,3], strides=[1,1], use_bias=False),
      layers.Conv2D(filters=conv3, kernel_size=[1,1], strides=[1,1], use_bias=False),
      layers.BatchNormalization(momentum=0.1),
      layers.ReLU(),
      layers.DepthwiseConv2D(kernel_size=[3,3], strides=[1,1], use_bias=False),
      layers.Conv2D(filters=conv4, kernel_size=[1,1], strides=[1,1], use_bias=False),
      layers.BatchNormalization(momentum=0.1),
      layers.ReLU(),
      layers.GlobalAveragePooling2D(),
      layers.Dense(num_labels)
    ])
    return model 

#********************************** MAIN BODY *****************************************
# DATASET PREPARATION -----------------------------------------------------------------
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

github_dir = pathlib.Path('./IotHomeworks')
if not github_dir.exists():
  !git clone https://github.com/alessandronicolini/IotHomeworks.git
code_path = "./IotHomeworks/HW2"

data_dir = pathlib.Path('data/mini_speech_commands')
if not data_dir.exists():
  tf.keras.utils.get_file(
      'mini_speech_commands.zip',
      origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
      extract=True,
      cache_dir='.', cache_subdir='data')
  
#lista di labels
labels=[]
for el in os.listdir("./data/mini_speech_commands"):
  if el!="README.md":
    labels.append(el)

#lista di training
training_list=[]
file=open(code_path+"/kws_train_split.txt")
for line in file:
  training_list.append('.'+line[1:-1])

#lista di validation
validation_list=[]
file=open(code_path+"/kws_val_split.txt")
for line in file:
  validation_list.append('.'+line[1:-1])

#lista di test
test_list=[]
file=open(code_path+"/kws_test_split.txt")
for line in file:
  test_list.append('.'+line[1:-1])

tot=len(training_list)+len(validation_list)+len(test_list)

if args.version == 'a':
  # original shape 49,10,1, ok for case a
  MFCC_OPTIONS = {'frame_length': 640, 'frame_step': 320, 'mfcc': True,'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,'num_coefficients': 10}
  shape = [49, 10, 1]

  model_options = {'conv1':128, 'conv2':64, 'conv3':32, 'conv4':32}

  ws_options = {'out_path':'./'+version+'_clustered', 
                'epochs':5, 
                'lr':1e-4, 
                'n_clusters':16, 
                'only_dense':False}

  ptq_options = {'out_path':'./Group11_kws_'+version,
                'q_type': 'int8int16'}

elif args.version == 'b':
  MFCC_OPTIONS = {'frame_length': 640, 'frame_step': 320, 'mfcc': True,'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,'num_coefficients': 10}
  shape = [49, 10, 1]

  model_options = {'conv1':128, 'conv2':64, 'conv3':32, 'conv4':32}

  ws_options = {'out_path':'./'+version+'_clustered', 
                'epochs':5, 
                'lr':1e-4, 
                'n_clusters':32, 
                'only_dense':False}

  ptq_options = {'out_path':'./Group11_kws_'+version}

elif args.version == 'c':
  # shape 39,10,1 good for total latency
  MFCC_OPTIONS = {'frame_length': 888, 'frame_step': 444, 'mfcc': True,'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,'num_coefficients': 10}
  shape = [35, 10, 1]

  model_options = {'conv1':128, 'conv2':128, 'conv3':64, 'conv4':16}

  ws_options = {'out_path':'./'+version+'_clustered', 
                'epochs':8, 
                'lr':1e-4, 
                'n_clusters':21, 
                'only_dense':False}

  ptq_options = {'out_path':'./Group11_kws_'+version}
                   
generator = SignalGenerator(labels, 16000, **MFCC_OPTIONS)     
train_ds = generator.make_dataset(training_list, True)
val_ds = generator.make_dataset(validation_list, False)
test_ds = generator.make_dataset(test_list, False)

n_batches = 200

# MODEL TRAINING ----------------------------------------------------------------------
model = get_model(**model_options)

model.summary()

# learning rate scheduler
learning_rate_fn = PolynomialDecay(
    initial_learning_rate=1e-3,
    decay_steps=3000,
    end_learning_rate=1e-5
    )

# compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_fn),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['sparse_categorical_accuracy'],
)

# callbacks
ckp_dir = "./checkpoint/"
try:
  os.mkdir(ckp_dir)
except FileExistsError:
  pass

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    ckp_dir, 
    monitor='val_sparse_categorical_accuracy', 
    verbose=0, 
    save_best_only=True,
    save_weights_only=False, 
    mode='max', 
    save_freq='epoch')

# fit model
EPOCHS = 30
history = model.fit(
    train_ds, 
    validation_data=val_ds,  
    epochs=EPOCHS,
    callbacks=[checkpoint_cb],
)

# TEST AND FP32 FILE CREATION --------------------------------------------------
# load and evaluate the best model
base_model = tf.keras.models.load_model(ckp_dir)
base_model.evaluate(test_ds, batch_size=32)
# save model and get the size of the 32 fp tf model
# SAVE MODELS-------------------------------------------------------------------
fp32_size = save_model(base_model, './'+version+'_fp32')
print('fp32 size: %.4f kB'%fp32_size)

# OPTIMIZATION -----------------------------------------------------------------
print("WEIGHTS CLUSTERING")
clustered_model, clustered_acc, clustered_size = weights_clustering(
    in_model=base_model, 
    **ws_options)
print('clustered accuracy: %.4f %%'%clustered_acc)
print('clustered size: %.4f kB'%clustered_size)
print()

# Post training quantization
print('POST TRAINING QUANTIZATION')
quant_acc, quant_size = pt_quantization(in_model=clustered_model, 
                                        **ptq_options)
print('accuracy of quantized model: %.4f %%'%quant_acc)
print('size of quantized model: %.4f kB'%quant_size)