//
// Created by mo on 18-3-6.
//

#include <net/if.h>
#include <sys/ioctl.h>
#include <openssl/md5.h>
#include <sstream>
#include <iomanip>
#include <map>
#include <fstream>
#include <iostream>

#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <boost/algorithm/string.hpp>
#include <string>

#include "mqttClient.h"



bool session = true;


const int Port = 1883;
const int KeepAlive = 60;
string messageID;
bool is_cps_ip_ready = false;

MQTTClient* mqtt_local_pub = new MQTTClient();

map<string, string> split(const string &s){
    map<string, string> retMap;
    typedef string::size_type string_size;
    string_size i = 0, j = 0;
    string key, value;
    while(j < s.size()){
        int flag = 0;
        while(j < s.size() && flag == 0){
            flag =1;
            if(s[j] != ':'){
                ++j;
                flag = 0;
            }
        }
        key = s.substr(i, j-i);
        j++;
        flag = 0;
        i = j;
        while(j < s.size() && flag == 0){
            flag =1;
            if(s[j] != ','){
                ++j;
                flag = 0;
            }
        }
        value = s.substr(i, j-i);
        j++;
        flag = 0;
        i = j;
        retMap[key] = value;
    }
    return retMap;
}

void MQTTClient::Init() {
    mosq_ = NULL;
    //libmosquitto 库初始化
    mosquitto_lib_init();
    string client_id = CLIENTID;
    string ip = ADDRESS;
    mosq_ = mosquitto_new(client_id.c_str(),session,NULL);
    if(!mosq_){
        printf("create client failed..\n");
        mosquitto_lib_cleanup();
    }
    //连接服务器
    while (mosquitto_connect(mosq_, ip.c_str(), Port, KeepAlive)) {
        fprintf(stderr, "host_name: local, Unable to connect.\n");
        usleep(5000000);
    }

    mosquitto_loop_start(mosq_);
}

void MQTTClient::pub_message(string buff) {
    string topic = TOPIC;
    mosquitto_publish(mosq_,NULL,topic.c_str(),strlen(buff.c_str()),buff.c_str(), 1, false);
}

