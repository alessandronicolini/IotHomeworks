# dataset
import argparse
import numpy as np
import pandas as pd
import os
import logging
import zlib

# metric class
from tensorflow.keras.metrics import Metric

# call backs
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

# models
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Flatten, Input, Concatenate, LSTM, Reshape, Dense
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse

# optimization
import tempfile
import tensorflow_model_optimization as tfmot

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True, help='model version')
args = parser.parse_args()

version = args.version

# DATASET CLASS ----------------------------------------------------------------
class WindowGenerator:

    def __init__(self, input_width, label_options, mean, std):
        self.input_width = input_width
        self.label_options = label_options
        self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])
        self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])


    def split_window(self, features):
        inputs = features[:, :-6, :]
        labels= features[:, -6:, :]
        num_labels = 6

        inputs.set_shape([None, self.input_width, 2])
        labels.set_shape([None, num_labels, 2])

        return inputs, labels


    def normalize(self, features):
        features = (features - self.mean) / (self.std + 1.e-6)

        return features


    def preprocess(self, features):
        inputs, labels = self.split_window(features)
        inputs = self.normalize(inputs)

        return inputs, labels


    def make_dataset(self, data, train):
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=input_width+6,
                sequence_stride=1,
                batch_size=32)

        ds = ds.map(self.preprocess)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds


# CUSTOM METRICS ---------------------------------------------------------------
class MaeTemp(Metric):

  def __init__(self, name="mae_T", **kwargs):
    super().__init__(name, **kwargs)
    self.total = self.add_weight('total', initializer='zeros') # accumulate the total error
    self.count = self.add_weight('count', initializer='zeros') # accumulate the number of elements 

  def reset_state(self):
    self.count.assign(tf.zero_like(self.count))
    self.total.assign(tf.zero_like(self.total))

    return

  def update_state(self, y_true, y_pred, sample_weight=None):
    # compute on how many elements you evaluate the error
    n_elements = 32*y_pred.shape[1] # 32*6
    
    # evaluate the error and accumulate first across batch elements and then 
    # across columns to get the total error related to temperature and humidity
    error = tf.abs(y_pred[:,:,0] - y_true[:,:,0]) # shape=[32, 6, 1] CONSIDER THE TEMPERATURE DATA
    error  = tf.reduce_sum(error, axis=0) # shape=[6,1]
    error = tf.reduce_sum(error, axis=0) # shape=[1]

    # update state variables
    self.total.assign_add(error)
    self.count.assign_add(n_elements)

    return
  

  def result(self):
    result = tf.math.divide_no_nan(self.total, self.count)

    return result


class MaeHum(Metric):

  def __init__(self, name="mae_H", **kwargs):
    super().__init__(name, **kwargs)
    self.total = self.add_weight('total', initializer='zeros') # accumulate the total error
    self.count = self.add_weight('count', initializer='zeros') # accumulate the number of elements 


  def reset_state(self):
    self.count.assign(tf.zero_like(self.count))
    self.total.assign(tf.zero_like(self.total))

    return


  def update_state(self, y_true, y_pred, sample_weight=None):
    # compute on how many elements you evaluate the error
    n_elements = 32*y_pred.shape[1]# 32*6
    
    # evaluate the error and accumulate first across batche elements and then 
    # across columns to get the total error related to temperature and humidity
    error = tf.abs(y_pred[:,:,1] - y_true[:,:,1]) # shape=[32, 6, 1] CONSIDER THE HUMIDITY DATA
    error  = tf.reduce_sum(error, axis=0) # shape=[6,1]
    error = tf.reduce_sum(error, axis=0) # shape=[1]

    # update state variables
    self.total.assign_add(error)
    self.count.assign_add(n_elements)

    return


  def result(self):
    result = tf.math.divide_no_nan(self.total, self.count)

    return result


# CUSTOM CHECKPOINT CLASS ------------------------------------------------------
class CustomCheckPoint(Callback):

  def __init__(self, chkp_dir):
    super(CustomCheckPoint, self).__init__()
    self.min_loss = np.inf
    self.min_mae_T = np.inf
    self.min_mae_H = np.inf
    self.mae_T_thr = 0.5
    self.mae_H_thr = 1.8
    self.chkp_dir = chkp_dir


  def on_epoch_end(self, epoch, logs=None):
    #current_loss = logs.get('val_loss')
    current_mae_T = logs.get('val_mae_T')
    current_mae_H = logs.get('val_mae_H')

    if (#current_loss <= self.min_loss and 
        current_mae_T < self.min_mae_T and 
        current_mae_H < self.min_mae_H):
        
        self.min_mae_T = current_mae_T
        self.min_mae_H = current_mae_H
        #self.min_loss = current_loss
    
        logs['best_epoch'] = epoch # Idk why but the value is appended, not overwritten
        self.model.save(self.chkp_dir)
        print("**checkpoint**")


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


#*********************** OPTIMIZATION FUNCTIONS ********************************
# WEIGHTS PRUNING---------------------------------------------------------------
def weights_pruning(in_model, out_path, epochs=5, lr=1e-3, polynomial=False, only_dense=True, in_sparsity=0.5, fin_sparsity=0.8):
    
    # create pruning scheduler
    if polynomial:
        prun_schedule=tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=in_sparsity,
            final_sparsity=fin_sparsity,
            begin_step=0,
            end_step=9200*epochs)
    else:
        prun_schedule=tfmot.sparsity.keras.ConstantSparsity(in_sparsity, 0)

    # create model for pruning
    if only_dense:

      def apply_pruning_to_dense(layer):
        if isinstance(layer, tf.keras.layers.Dense):
          return tfmot.sparsity.keras.prune_low_magnitude(
              layer, 
              pruning_schedule=prun_schedule,
              block_size = (1, 1),
              block_pooling_type = 'AVG')                                      
        return layer

      model_for_pruning = tf.keras.models.clone_model(
          in_model, 
          clone_function=apply_pruning_to_dense)

    else:

      model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
          in_model,
          pruning_schedule = prun_schedule,
          block_size = (1, 1),
          block_pooling_type = 'AVG'
      )
    
    # compile model
    model_for_pruning.compile(optimizer=Adam(learning_rate=lr),
                              loss=mse,
                              metrics=[MaeTemp(), MaeHum()])
    # pruning callbacks
    logdir = tempfile.mkdtemp()
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep(),
                 tfmot.sparsity.keras.PruningSummaries(log_dir=logdir)]
    
    # fit model
    model_for_pruning.fit(train_ds,
                          batch_size=32, 
                          epochs=epochs, 
                          validation_data=val_ds,
                          callbacks=callbacks)
    
    # pruned model evaluation
    _, mae_t, mae_h = model_for_pruning.evaluate(test_ds, verbose=0)
    
    # prepare for export and save
    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    zip_size = save_model(model_for_export, out_path)
                           
    return model_for_export, mae_t, mae_h, zip_size


# WEIGHTS CLUSTERING------------------------------------------------------------
def weights_clustering(in_model, out_path, epochs=5, lr=1e-3, n_clusters=4, only_dense=True):
    
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
    clustered_model.compile(optimizer=Adam(learning_rate=lr),
        loss=mse,
        metrics=[MaeTemp(), MaeHum()])
    
    # fit model
    clustered_model.fit(train_ds,
                        batch_size=32, epochs=epochs, validation_data=val_ds)
    
    # clustered model evaluation
    _, mae_t, mae_h = clustered_model.evaluate(test_ds, verbose=0)
    
    # prepare for export and save
    model_for_export = tfmot.clustering.keras.strip_clustering(clustered_model)
    zip_size = save_model(model_for_export, out_path)
    
    return model_for_export, mae_t, mae_h, zip_size


# POST TRAINING QUANTIZATION-----------------------------------------------------
def get_predictions(interpreter, test_dataset):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    predictions = []

    for batch in test_dataset:
    
        for test_sample in batch[0]:
            test_sample = np.expand_dims(test_sample, axis=0).astype(np.float32)
            interpreter.set_tensor(input_index, test_sample)
            interpreter.invoke()
            output = interpreter.get_tensor(output_index)
            predictions.append(output[0])

    predictions = np.array(predictions)

    return predictions 


def unbatch_targets(test_dataset):
    dataset = test_dataset.unbatch()
    dataset = list(dataset.as_numpy_iterator())
    targets = np.array(dataset)
    targets = targets[:,1,:,:]
    
    return targets


def evaluate_mae(interpreter, test_dataset):
    predictions = get_predictions(interpreter, test_dataset)
    targets = unbatch_targets(test_dataset)
    errorT = []
    errorH = []
    for prediction, target in zip(predictions, targets):
        errT = np.abs(target[:,0]-prediction[:,0])
        errH = np.abs(target[:,1]-prediction[:,1])
        errT = np.average(errT, axis=0)
        errH = np.average(errH, axis=0)
        errorT.append(errT)
        errorH.append(errH)
    errorT = np.array(errorT)
    errorH = np.array(errorH)
    mae_t = np.average(errorT)
    mae_h = np.average(errorH)

    return mae_t, mae_h
        
           
def pt_quantization(in_model, out_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(in_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    quant_tflite_model = converter.convert()

    # compressed file
    compressed_model = zlib.compress(quant_tflite_model)
        
    # .zip format   
    with open(out_path+'.tflite.zlib', 'wb') as f:
        f.write(compressed_model)
    
    
    interpreter = tf.lite.Interpreter(model_content=quant_tflite_model)
    interpreter.allocate_tensors()
    
    mae_t, mae_h = evaluate_mae(interpreter, test_ds)
    zip_size = os.path.getsize(out_path+'.tflite.zlib')/1024
    
    return mae_t, mae_h, zip_size


#******************************* BASE MODEL ************************************
# INITIALIZATION STEP ----------------------------------------------------------
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')
csv_path, _ = os.path.splitext(zip_path)
df = pd.read_csv(csv_path)

column_indices = [2, 5]
columns = df.columns[column_indices]
data = df[columns].values.astype(np.float32)

n = len(data)
train_data = data[0:int(n*0.7)]
val_data = data[int(n*0.7):int(n*0.9)]
test_data = data[int(n*0.9):]

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

input_width = 6
LABEL_OPTIONS = 6 #args.labels


# MAKE THE DATASET -------------------------------------------------------------
generator = WindowGenerator(input_width, LABEL_OPTIONS, mean, std)
train_ds = generator.make_dataset(train_data, True)
val_ds = generator.make_dataset(val_data, False)
test_ds = generator.make_dataset(test_data, False)


# INITIALIZE AND FIT MODEL -----------------------------------------------------
# initialize model
input = Input(shape=(6, 2))
x = Flatten()(input)
x = Dense(units=8, activation='relu')(x)
x = Dense(units=12, activation='relu')(x)
x = Dense(units=12)(x)
x = Reshape(target_shape=[6,2])(x)
model = Model(inputs=input, outputs=x, name='base_model')

print("*"*70)
print("BASE MODEL STRUCTURE")
print("*"*70)

model.summary()

# compile model
model.compile(optimizer=Adam(learning_rate=5e-4),
              loss=mse,
              metrics=[MaeTemp(), MaeHum()])

# callbacks
chkp_dir = "./checkpoint/best"  # current configuration best model dir
chkp_cb = CustomCheckPoint(chkp_dir)

epochs = 30

# fit model

print('\n\n')
print("*"*70)
print("BASE MODEL TRAINING")
print("*"*70)

history = model.fit(train_ds,
                    batch_size=32,
                    epochs=epochs,  # first training
                    validation_data=val_ds,
                    callbacks=[chkp_cb])

# TEST BASE MODEL --------------------------------------------------------------
# evaluate the best found model
base_model = tf.keras.models.load_model(
    './checkpoint/best',
    custom_objects={'MaeTemp':MaeTemp, 'MaeHum':MaeHum})

loss, mae_t, mae_h = base_model.evaluate(test_ds, batch_size=32, verbose=0)

#print("test loss: %.3f"%loss)
print("\ntest MAE T: %.3f"%mae_t)
print("test MAE H: %.3f"%mae_h)


base_model.save('base_model')
converter = tf.lite.TFLiteConverter.from_saved_model("base_model")
base_tflite_model = converter.convert()
compressed_base_model = zlib.compress(base_tflite_model)

with open('base_tflite_model.tflite.zlib', 'wb') as f:
    f.write(compressed_base_model)

size = os.path.getsize('base_tflite_model.tflite.zlib')/1024    
print("size of base tflite model w/o optimization: %.3f kB"%size)

#**************************** MODEL OPTIMIZATION *******************************
print('\n\n')
print("*"*70)
if version =='a':
  print("BASE MODEL OPTIMIZATION version a:\n- weights clustering\n- post trainng quantization")
elif version == 'b':
  print("BASE MODEL OPTIMIZATION version b:\n- weights pruning\n- weights clustering\n- post trainng quantization")
print("*"*70)

if version == 'a':
    #maeT < 0.5, maeH < 1.8, size < 2kB
    #weight clustering and quantization 

    print("\n\nWEIGHTS CLUSTERING RETRAINING\n")
    clustered_model, c_mae_t, c_mae_h, c_size = weights_clustering(
        base_model, 
        './clustered', 
        epochs=8, 
        n_clusters=18, 
        lr=1e-4, 
        only_dense=True)
    print()
    print('mae t %.3f'%c_mae_t)
    print('mae h %.3f'%c_mae_h)
    print('size %.3f kB'%c_size)

    print("\n\nPOST TRAINING QUANTIZATION\n")
    q_mae_t, q_mae_h, q_size = pt_quantization(clustered_model, './Group11_th_'+version)
    print()
    print('mae t %.3f'%q_mae_t)
    print('mae h %.3f'%q_mae_h)
    print('size %.3f kB'%q_size)

elif version == 'b':
    # maeT < 0.6, maeH < 1.8, size < 1.7kB
    # pruning, weight clustering and quantization

    print("\n\nPRUNING RETRAINING\n")
    pruned_model, p_mae_t, p_mae_h, p_size = weights_pruning(
        base_model,
        './pruned', 
        polynomial=True, 
        lr=1e-4, 
        epochs=4,
        in_sparsity=0.2,
        fin_sparsity=0.5)
    print('\nmae t %.3f'%p_mae_t)
    print('mae h %.3f'%p_mae_h)
    print('size %.3f kB'%p_size)

    print("\n\nWEIGHTS CLUSTERING RETRAINING\n")
    clustered_model, c_mae_t, c_mae_h, c_size = weights_clustering(
        pruned_model, 
        './clustered', 
        epochs=4, 
        n_clusters=12, 
        lr=1e-4, 
        only_dense=True)
    print('\nmae t %.3f'%c_mae_t)
    print('mae h %.3f'%c_mae_h)
    print('size %.3f kB'%c_size)

    print("\n\nPOST TRAINING QUANTIZATION\n")
    q_mae_t, q_mae_h, q_size = pt_quantization(clustered_model, './Group11_th_'+version)
    print()
    print('mae t %.3f'%q_mae_t)
    print('mae h %.3f'%q_mae_h)
    print('size %.3f kB'%q_size)