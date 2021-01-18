import paho.mqtt.client as PahoMQTT
import time
from BasicClient import basicClient
import tensorflow as tf

class modelClient(basicClient):
    
    def __init__(self, clientID, subscribe_topics, publish_topic, model_path):
        
        super().__init__(clientID, subscribe_topics, publish_topic)
        
        # model attributes
        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def myOnMessageReceived(self, paho_mqtt , userdata, msg):
        print(msg.payload)
        
        # leggi il payload del messaggio in formato SenML+JSON e prendi gli mfcc
        
        # leggi anche l'indice dell'elemento ricevuto
        
        # fai la predizione
        
        # manda indietro i valori dei neuroni dell'ultimo layer in formato JSON insieme all'indice corrispondente
   