import paho.mqtt.client as PahoMQTT
import time
from BasicClient import basicClient
import tensorflow as tf

class raspClient(basicClient):
    
    def __init__(self, clientID, subscribe_topics, publish_topic, length_testset):
        
        super().__init__(clientID, subscribe_topics, publish_topic)
        self.prediction_dict = {i:[] for i in range(length_testset)}
        
    def myOnMessageReceived(self, paho_mqtt , userdata, msg):
        print(msg.payload)
        # leggi le predicioni rivevute in formato JSON
        
        # leggi l'indice dell'elemento predetto
        
        # appendi la predizione alla chiave corrispondente nel dizionario
        
        # se arriva il messaggio di stop calcoliamo l'accuracy
        