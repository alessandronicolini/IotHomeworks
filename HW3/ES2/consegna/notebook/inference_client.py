import paho.mqtt.client as PahoMQTT
import time
from basicMQTT import basicMQTT
import tensorflow as tf
import json
import numpy as np
import argparse
import pathlib
import os

class inferenceMQTT(basicMQTT):

    def __init__(self, clientID, subscribe_topics, publish_topic, model_path, QoS):

        super().__init__(clientID, subscribe_topics, publish_topic, QoS)

        # model attributes
        self.interpreter = tf.lite.Interpreter(model_path)
        self.stop_flag = False
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


    def myOnMessageReceived(self, paho_mqtt , userdata, msg):

        # convert input message from binary string -> string -> dictionary
        str_msg = msg.payload.decode()
        dict_msg = json.loads(str_msg)

        # get sample index
        resource_id = dict_msg['e'][0]['n'].split('_')
        sample_idx = int(resource_id[1])

        # get sample preprocessed
        mfcc = tf.convert_to_tensor(dict_msg['e'][0]['vd'])

        # make prediction
        mfcc = np.expand_dims(mfcc, axis=0).astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], mfcc)
        self.interpreter.invoke()
        logits = self.interpreter.get_tensor(self.output_details[0]['index'])
        probs = tf.nn.softmax(logits)

        # make out dictionay
        to_rasp_dict = {'probs': probs.numpy().squeeze().tolist(), 'idx':sample_idx}
        json_to_rasp = json.dumps(to_rasp_dict)

        # send output layer to the raspberry
        self.myPublish(self._publish_topic, json_to_rasp)

        # if it was the last message stop the inference client
        if len(resource_id) == 3:
            self.stop_flag = True



    def myPublish(self, topic, message):
        self._paho_mqtt.publish(self._publish_topic, message, self._QoS)


if __name__ == '__main__':

    # read the input argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='model path')
    args = parser.parse_args()
    model_path = args.model

    # define subsscribed and publish topics
    model_n = model_path.split('.')[0][-1]
    subscribed_to, publish_on = ['/Group11IoTHW32020/prep_sample'], '/Group11IoTHW32020/model/'+model_n

    # initialize and start the inference client
    inf_options = {
        'clientID': 'model' + model_n,
        'subscribe_topics': subscribed_to,
        'publish_topic': publish_on,
        'model_path': model_path,
        'QoS': 0
    }

    inference_client = inferenceMQTT(**inf_options)
    inference_client.start()

    while not inference_client.stop_flag:
        pass

    time.sleep(5)
    inference_client.stop()
