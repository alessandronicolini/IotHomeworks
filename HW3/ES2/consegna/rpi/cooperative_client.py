import paho.mqtt.client as PahoMQTT
import time
from basicMQTT import basicMQTT
import tensorflow as tf
import json
import numpy as np
from dataset import SignalGenerator
import pathlib
import os
import datetime
from datetime import timezone


class coopMQTT(basicMQTT):

    def __init__(self, clientID, subscribe_topics, publish_topic, n_models, len_dataset, QoS, n_labels):

        super().__init__(clientID, subscribe_topics, publish_topic, QoS)
        self._n_models = n_models
        self.stop_flag = False
        self._preds_dict = dict()
        self.running_corrects = 0
        self._n_labels = n_labels
        self._len_dataset = len_dataset


    def myPublish(self, topic, message, sample_idx, sample_label):
        # create the element which will contains the predictions
        self._preds_dict[sample_idx] = [sample_label, np.zeros(self._n_labels), 0]

        # send the message to the inference clients
        self._paho_mqtt.publish(self._publish_topic, message, self._QoS)


    def myOnMessageReceived(self, paho_mqtt , userdata, msg):

        # convert input message from binary string -> string -> dictionary
        str_msg = msg.payload.decode()
        dict_msg = json.loads(str_msg)

        # append the prediction to the respective dictionary element
        sample_idx = dict_msg['idx']
        preds = np.array(dict_msg['probs'])
        self._preds_dict[sample_idx][1] += preds
        self._preds_dict[sample_idx][2] += 1

        # check if you can make the cooperative prediction
        if self._preds_dict[sample_idx][2] == self._n_models:
            if (sample_idx+1)%40==0 :
                print("progress : %.2f%%" %((sample_idx+1)/8))

            # cooperative prediction
            c_pred = np.argmax(self._preds_dict[sample_idx][1])
            if self._preds_dict[sample_idx][0] == c_pred:
                self.running_corrects += 1

            # stop the rasp client
            if sample_idx == self._len_dataset-1:
                self.stop_flag = True


            # delete the already predicted element from the dictionary
            del self._preds_dict[sample_idx]


if __name__ == '__main__':

    # DATASET PREPARATION -----------------------------------------
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # download dataset
    data_dir = pathlib.Path('data/mini_speech_commands')
    if not data_dir.exists():
      tf.keras.utils.get_file(
          'mini_speech_commands.zip',
          origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
          extract=True,
          cache_dir='.', cache_subdir='data')

    # lista di test
    test_list=[]
    file=open("./kws_test_split.txt")
    for line in file:
      test_list.append('.'+line[1:-1])

    # lista di labels - da replicare in EX1
    labels = open('labels.txt').readlines()[0].split()

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
    test_ds = generator.make_dataset(test_list, False)

    # COOPERATIVE CLIENT (RASP)----------------------------------------------------------------
    timestamp = int(datetime.datetime.now(timezone.utc).timestamp())
    device_name = 'rasp'

    subscribed_to, publish_on = ['/Group11IoTHW32020/model/1',
                                 '/Group11IoTHW32020/model/2',
                                 '/Group11IoTHW32020/model/3'], '/Group11IoTHW32020/prep_sample'

    # initialize and start the cooperative client
    coop_options = {
        'clientID': 'coopClient',
        'subscribe_topics': subscribed_to,
        'publish_topic': publish_on,
        'QoS': 0,
        'n_models': 3,
        'len_dataset': len(test_ds),
        'n_labels': len(labels)
    }

    coopClient = coopMQTT(**coop_options)
    coopClient.start()

    # send samples mfccs to the inference clients
    for i, (mfcc, label) in enumerate(test_ds):

        resource_id = "sample_"+str(i)
        if i == len(test_ds)-1:
            resource_id += "_last"

        senml_msg = {
            "bn" : device_name,
            "bt" : timestamp,
            "e" :[{"n":resource_id, "u":"/", "t":0, "vd":mfcc.numpy().tolist()}]
            }

        senml_msg = json.dumps(senml_msg)
        coopClient.myPublish(publish_on, senml_msg, i, label.numpy())

        # tune the sleep parameter:
        # each client must be able to do the inference before receiving the next sample
        time.sleep(0.04)

    while not coopClient.stop_flag:
        pass

    print('Accuracy: %.3f %%'%(coopClient.running_corrects/len(test_ds)*100))
    coopClient.stop()
