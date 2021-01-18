import paho.mqtt.client as PahoMQTT
import time

class basicClient:
    
    def __init__(self, clientID, subscribe_topics, publish_topic):

        self.clientID = clientID

        # create an instance of paho.mqtt.client
        self._paho_mqtt = PahoMQTT.Client(clientID, False) 

        # register the callbacks
        self._paho_mqtt.on_connect = self.myOnConnect
        self._paho_mqtt.on_message = self.myOnMessageReceived
        
        self._subscribe_topics = subscribe_topics
        self._publish_topic = publish_topic
        self.messageBroker = 'mqtt.eclipseprojects.io'

            
    def start (self):
        #manage connection to broker
        self._paho_mqtt.connect(self.messageBroker, 1883, keepalive=5)
        self._paho_mqtt.loop_start()
        # subscribe for the topics
        for topic in self._subscribe_topics:
            self._paho_mqtt.subscribe(topic, 2)

                
    def stop (self):
        for topic in self._subscribe_topics:
            self._paho_mqtt.unsubscribe(topic)
        self._paho_mqtt.loop_stop()
        self._paho_mqtt.disconnect()
            
    def myPublish(self, topic, message):
        # publish a message with a certain topic
        self._paho_mqtt.publish(self._publish_topic, message, 2)

    def myOnConnect(self, paho_mqtt, userdata, flags, rc):
        print("Connected to %s with result code: %d" % (self.messageBroker, rc))

    def myOnMessageReceived(self, paho_mqtt , userdata, msg):
        pass
        """# A new message is received
        print("Topic:'" + msg.topic+"', QoS: '"+str(msg.qos)+"' Message: '"+str(msg.payload) + "'")
        if str(msg.payload) == "b'STOP'":
            print('stopped message received: stopping...')
            self.stop()"""