import os
import pathlib
import zlib
import numpy as np
import tempfile
import argparse

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from keras.optimizers.schedules import PolynomialDecay
import keras

import tensorflow_model_optimization as tfmot

#dataset.py must be included in the same folder as the script
from dataset import SignalGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=int, required=True, help='model version')
args = parser.parse_args()
version = args.version

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

    with open(out_path+'.tflite', 'wb') as f:
        f.write(quant_tflite_model)

    print("model saved to: "+out_path+".tflite")
    interpreter = tf.lite.Interpreter(model_content=quant_tflite_model)
    interpreter.allocate_tensors()

    accuracy = get_accuracy(interpreter, test_ds)


    return accuracy, quant_tflite_model

# DATASET CREATION

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

data_dir = pathlib.Path('data/mini_speech_commands')
if not data_dir.exists():
  tf.keras.utils.get_file(
      'mini_speech_commands.zip',
      origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
      extract=True,
      cache_dir='.', cache_subdir='data')

#lista di training
training_list=[]
file=open("kws_train_split.txt")
for line in file:
  training_list.append('.'+line[1:-1])

#lista di validation
validation_list=[]
file=open("kws_val_split.txt")
for line in file:
  validation_list.append('.'+line[1:-1])

# lista di test
test_list=[]
file=open("kws_test_split.txt")
for line in file:
  test_list.append('.'+line[1:-1])

# lista di labels
labels = open('labels.txt').readlines()[0].split()
print(labels)


MFCC_OPTIONS = {
    'frame_length': 640,
    'frame_step': 320,
    'mfcc': True,
    'lower_frequency': 20,
    'upper_frequency': 4000,
    'num_mel_bins': 40,
    'num_coefficients': 10
}

# make test dataset
generator = SignalGenerator(labels, 16000, **MFCC_OPTIONS)
train_ds = generator.make_dataset(training_list, True)
val_ds = generator.make_dataset(validation_list, False)
test_ds = generator.make_dataset(test_list, False)


shape = [49, 10, 1]
if version == 1:
    model = models.Sequential([
      layers.Input(shape=shape),
      layers.Conv2D(filters=256, kernel_size=[3,3], strides=[2,1], use_bias=False),
      layers.BatchNormalization(momentum=0.1),
      layers.ReLU(),
      layers.DepthwiseConv2D(kernel_size=[3,3], strides=[1,1], use_bias=False),
      layers.Conv2D(filters=256, kernel_size=[1,1], strides=[1,1], use_bias=False),
      layers.BatchNormalization(momentum=0.1),
      layers.ReLU(),
      layers.DepthwiseConv2D(kernel_size=[3,3], strides=[1,1], use_bias=False),
      layers.Conv2D(filters=128, kernel_size=[1,1], strides=[1,1], use_bias=False),
      layers.BatchNormalization(momentum=0.1),
      layers.ReLU(),
      layers.DepthwiseConv2D(kernel_size=[3,3], strides=[1,1], use_bias=False),
      layers.Conv2D(filters=64, kernel_size=[1,1], strides=[1,1], use_bias=False),
      layers.BatchNormalization(momentum=0.1),
      layers.ReLU(),
      layers.GlobalAveragePooling2D(),
      layers.Dense(8)
    ])
elif version == 2:
    model = models.Sequential([
        layers.Input(shape=shape),
        layers.Reshape((49, 10), input_shape=shape),
        layers.LSTM(units=512),
        layers.Flatten(),
        layers.Dense(8)])
elif version == 3:
    model = models.Sequential([
      layers.Input(shape=shape),
      layers.Conv2D(filters=512, kernel_size=[3,3], strides=[2,1], use_bias=False),
      layers.BatchNormalization(momentum=0.1),
      layers.ReLU(),
      layers.DepthwiseConv2D(kernel_size=[3,3], strides=[1,1], use_bias=False),
      layers.Conv2D(filters=512, kernel_size=[1,1], strides=[1,1], use_bias=False),
      layers.BatchNormalization(momentum=0.1),
      layers.ReLU(),
      layers.DepthwiseConv2D(kernel_size=[3,3], strides=[1,1], use_bias=False),
      layers.Conv2D(filters=128, kernel_size=[1,1], strides=[1,1], use_bias=False),
      layers.BatchNormalization(momentum=0.1),
      layers.ReLU(),
      layers.GlobalAveragePooling2D(),
      layers.Dense(8)
    ])
else :
    print("invalid input value!")


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
    callbacks=[checkpoint_cb],)

# load and evaluate the best model
base_model = tf.keras.models.load_model(ckp_dir)
acc = base_model.evaluate(test_ds, batch_size=32, return_dict=True)['sparse_categorical_accuracy']
print()
print("sparse categorical accuracy on test set : "+str(acc))

### OPTIMIZATIONS

out_path = "model_"+str(version)

quant_acc, quant_model = pt_quantization(in_model=base_model, out_path=out_path, q_type='float16')
print('accuracy of quantized model: %.4f %%'%quant_acc)
