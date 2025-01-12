//
//  main.cpp
//  UltraFaceTest
//
//  Created by vealocia on 2019/10/17.
//  Copyright © 2019 vealocia. All rights reserved.
//

#include "UltraFace.hpp"
#include "mqttClient.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <zconf.h>
#include <sys/time.h>


unsigned long long get_timestamp() {
    struct timeval tv;
    if (gettimeofday(&tv, 0) == -1) printf("Failed to time stamp.\n");
    unsigned long long time_stamp = tv.tv_sec * 1000 + tv.tv_usec / 1000;
    return time_stamp;
}

int main(int argc, char **argv) {
    if (argc <= 3) {
        fprintf(stderr, "Usage: %s <ncnn bin> <ncnn param> [image files...]\n", argv[0]);
        return 1;
    }

    std::string bin_path = argv[1];
    std::string param_path = argv[2];
//    cv::namedWindow("UltraFace");
    UltraFace ultraface(bin_path, param_path, 320, 1, 0.6); // config model input

    cv::VideoCapture cap(0);
    cv::Mat frame;
    mqtt_local_pub->Init();
    cap.read(frame);
    float left_limit = frame.cols / 3.0;
    float right_limit = frame.cols / 3.0 * 2;

    int left_num = 0;
    int middle_num = 0;
    int right_num = 0;

//    unsigned long long last_pub_mqtt_time = get_timestamp();

    while (true) {
        left_num = 0;
        middle_num = 0;
        right_num = 0;
        cap.read(frame);
        std::vector<FaceInfo> face_info;
        std::vector<FaceInfo> person_info;
        unsigned long long start = get_timestamp();
        ncnn::Mat inmat = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);
        ultraface.detect(inmat, person_info, face_info);
        unsigned long long stop = get_timestamp();
        std::cout << "detect time: " << stop-start << " ms" << std::endl;

        for (unsigned int i = 0; i < face_info.size(); i++) {
            auto face = face_info[i];
            float center_x = (face.x1 + face.x2)/2;
            if (center_x < left_limit) {
                left_num++;
            } else if (center_x < right_limit) {
                middle_num++;
            } else {
                right_num++;
            }
            cv::Point pt1(face.x1, face.y1);
            cv::Point pt2(face.x2, face.y2);
            cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);
        }

        for (unsigned int i = 0; i < person_info.size(); i++) {
            auto person = person_info[i];
            float center_x = (person.x1 + person.x2)/2;
            if (center_x < left_limit) {
                left_num++;
            } else if (center_x < right_limit) {
                middle_num++;
            } else {
                right_num++;
            }
            cv::Point pt1(person.x1, person.y1);
            cv::Point pt2(person.x2, person.y2);
            cv::rectangle(frame, pt1, pt2, cv::Scalar(255, 0, 0), 2);
        }

//        if (get_timestamp() - last_pub_mqtt_time > 1000) {
//            last_pub_mqtt_time = get_timestamp();
//            string pub_message = to_string(left_num) + "," + to_string(middle_num) + "," + to_string(right_num);
//            mqtt_local_pub->pub_message(pub_message);
//        }

//        cv::imshow("UltraFace", frame);
//        cv::waitKey(33);
    }


    return 0;
}
