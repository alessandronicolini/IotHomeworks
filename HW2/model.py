from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tf



class MultiOutputModel():

    def build_temperature_branch(self, inputs):

        x = Conv2D(filters=64, kernel_size=(2,2), activation="relu")(inputs)
        x = Flatten()(x)
        x = Dense(units=64, activation="relu")(x)
        x = Dense(units=6)(x)
        x = tf.reshape(x, (-1, 6, 1)) 
        return x

    def build_humidity_branch(self, inputs):

        x = Conv2D(filters=64, kernel_size=(2,2), activation="relu")(inputs)
        x = Flatten()(x)
        x = Dense(units=64, activation="relu")(x)
        x = Dense(units=6)(x)
        x = tf.reshape(x, (-1, 6, 1))
        return x


    def assemble_full_model(self):

        inputs = Input(shape=(6,2,1))
        temperature_branch = self.build_temperature_branch(inputs)
        humidity_branch = self.build_humidity_branch(inputs)
        outputs = tf.keras.layers.Concatenate(axis=2)([temperature_branch, humidity_branch])
        model = Model(inputs=inputs, outputs = outputs)
        return model
    
model = MultiOutputModel().assemble_full_model()
