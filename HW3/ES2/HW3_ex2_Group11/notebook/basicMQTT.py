import paho.mqtt.client as PahoMQTT
import time

class basicMQTT:
    
    def __init__(self, clientID, subscribe_topics, publish_topic, QoS):

        self.clientID = clientID

        # create an instance of paho.mqtt.client
        self._paho_mqtt = PahoMQTT.Client(clientID, True, None, PahoMQTT.MQTTv31) # clean_session = False

        # register the callbacks
        self._paho_mqtt.on_connect = self.myOnConnect
        self._paho_mqtt.on_message = self.myOnMessageReceived
        
        self._subscribe_topics = subscribe_topics
        self._publish_topic = publish_topic
        self.messageBroker = 'mqtt.eclipseprojects.io'
        self._QoS = QoS
            
    def start (self):
        #manage connection to broker
        self._paho_mqtt.connect(self.messageBroker, 1883) # keepalive = 5
        self._paho_mqtt.loop_start()
        
        # subscribe for the topics
        for topic in self._subscribe_topics:
            self._paho_mqtt.subscribe(topic, 2)

                
    def stop (self):
        # unsubscribe fro all previoulsy subscriberd topics
        for topic in self._subscribe_topics:
            self._paho_mqtt.unsubscribe(topic)
        print(self.clientID +' :stopped and disconnected')
        
        # stop and disconnect
        self._paho_mqtt.loop_stop()
        self._paho_mqtt.disconnect()
    
    
    def myOnConnect(self, paho_mqtt, userdata, flags, rc):
        print("Connected to %s with result code: %d" % (self.messageBroker, rc))
    
    
    def myPublish(self, topic, message):
        pass

        
    def myOnMessageReceived(self, paho_mqtt , userdata, msg):
        pass
        
