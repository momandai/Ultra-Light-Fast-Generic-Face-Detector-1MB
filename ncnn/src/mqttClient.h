//
// Created by mo on 18-3-6.
//

#ifndef MQTT_SUB_CLASS_MQTTCLIENT_H
#define MQTT_SUB_CLASS_MQTTCLIENT_H
#include <mosquitto.h>
#include <iostream>

#include <opencv2/core/core.hpp>

using namespace std;

//#define ADDRESS     "tcp://test.mosquitto.org:1883"
#define ADDRESS     "47.97.194.171"
#define CLIENTID    "30d690cb-09e4-4921-8f38-e79711cfb0ff"
#define TOPIC       "position"

class MQTTClient{
public:
    void Init();
    void pub_message(string buff);

private:
    struct mosquitto *mosq_;
};


extern MQTTClient* mqtt_local_pub;


#endif //MQTT_SUB_CLASS_MQTTCLIENT_H
