import os
import glob
import re
import json
import random
import cv2
import numpy as np
from vision.ssd.config import fd_config
from vision.ssd.config.fd_config import define_img_size

define_img_size(320)


class CaltechDataset:
    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False, label_file=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.datalist = []
        self.train_datalist = []
        self.test_datalist = []
        self.train_list_name = "train_data_list.json"
        self.test_list_name = "test_data_list.json"
        self.class_names = ('BACKGROUND', 'person')
        self.is_test = is_test
        if os.path.exists(os.path.join(root, self.train_list_name)) and os.path.exists(os.path.join(root, self.test_list_name)):
            self.train_datalist = json.load(open(os.path.join(root, self.train_list_name)))
            self.test_datalist = json.load(open(os.path.join(root, self.test_list_name)))
        else:
            annotations = json.load(open(os.path.join(root, "annotations.json")))
            for set_num in annotations.keys():
                if set_num != "set00":
                    continue
                for V_num in annotations[set_num].keys():
                    for frame_num in annotations[set_num][V_num]['frames'].keys():
                        data = annotations[set_num][V_num]['frames'].get(frame_num)
                        image_name = set_num + "_" + V_num + "_" + frame_num + ".png"
                        image_path = os.path.join(root, 'images', set_num, image_name)
                        if not os.path.exists(image_path):
                            continue
                        boxes = []
                        labels = []
                        for datum in data:
                            label = datum['lbl']
                            if label != 'person':
                                continue
                            x, y, w, h = datum['pos']
                            boxes.append([x, y, x+w, y+h])
                            labels.append(1)
                        self.datalist.append({"image_path": image_path, "boxes": boxes, "labels": labels})

            random.shuffle(self.datalist)
            part = int(len(self.datalist) * 0.75)
            self.train_datalist = self.datalist[0:part]
            self.test_datalist = self.datalist[part:]
            json.dump(self.train_datalist, open(os.path.join(root, self.train_list_name), 'w'))
            json.dump(self.test_datalist, open(os.path.join(root, self.test_list_name), 'w'))

        print("dataset laoded!")
        # image_paths = glob.glob('/home/test/data/*/*.png')
        # for image_path in image_paths:
        #     img_name = os.path.basename(image_path)
        #     set_name = re.search('(set[0-9]+)', img_name).groups()[0]
        #     video_name = re.search('(V[0-9]+)', img_name).groups()[0]
        #     n_frame = re.search('_([0-9]+)\.png', img_name).groups()[0]
        #     data = annotations[set_name][video_name]['frames'].get(n_frame)


    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def show_image_and_label(self, index):
        index = index % len(self.train_datalist)
        data = self.train_datalist[index]
        image_path = data["image_path"]
        boxes = data["boxes"]
        image = cv2.imread(image_path)
        for box in boxes:
            x0, y0, x1, y1 = [int(v) for v in box]
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 1)
        cv2.imshow("image", image)
        cv2.waitKey(-1)

    def __len__(self):
        if self.is_test:
            return len(self.test_datalist)
        else:
            return len(self.train_datalist)

    def __getitem__(self, index):
        if not self.is_test:
            index = index % len(self.test_datalist)
            data = self.test_datalist[index]
        else:
            index = index % len(self.train_datalist)
            data = self.train_datalist[index]
        image_path = data["image_path"]
        boxes = np.array(data["boxes"])
        labels = np.array(data["labels"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)

        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels


if __name__ == '__main__':
    print(fd_config.priors)
    dataset = CaltechDataset("caltech_data")
    dataset.__getitem__(334)
    dataset.show_image_and_label(334)
