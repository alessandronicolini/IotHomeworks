import paho.mqtt.client as PahoMQTT
import time
from basicMQTT import basicMQTT
import tensorflow as tf
import json
import numpy as np
from collections import Counter

class coopMQTT(basicMQTT):
    
    def __init__(self, clientID, subscribe_topics, publish_topic, n_models, len_dataset, QoS, n_labels):
        
        super().__init__(clientID, subscribe_topics, publish_topic, QoS)
        self._n_models = n_models
        self._preds_dict = dict()
        self.running_corrects = 0
        self._n_labels = n_labels
        self._len_dataset = len_dataset
        self.last_received = False
        
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
            print(sample_idx)
            
            # cooperative prediction
            """c_pred = Counter(self._preds_dict[sample_idx][1])
            c_pred = list(c_pred.items())
            c_pred.sort(key= lambda x:x[1], reverse=True)
            c_pred = c_pred[0][0]"""
            c_pred = np.argmax(self._preds_dict[sample_idx][1])
            if self._preds_dict[sample_idx][0] == c_pred:
                self.running_corrects += 1
            
            if sample_idx == self._len_dataset-1:
                self.last_received = True
                         
            # delete the already predicted element from the dictionary
            del self._preds_dict[sample_idx]