import logging
import os
import pathlib
import json

import cv2
import numpy as np


class COCODataset:

    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False, label_file=None):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        if is_test:
            data_list_file = self.root / "test_datalist.json"
        else:
            data_list_file = self.root / "train_datalist.json"

        self.data_list = json.load(open(data_list_file))
        print("{0} loaded, total labels: {1}".format(self.root / "test_datalist.json", len(self.data_list)))

        self.class_names = ('BACKGROUND', 'person', 'face')

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        data = self.data_list[index]
        image_id = data['image_file']
        boxes = np.array(data['boxes'], dtype=np.float32)
        labels = np.array(data['labels'], dtype=np.int64)
        image = cv2.imread(image_id)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def __len__(self):
        return len(self.data_list)

    def show_image_and_labels(self, index):
        data = self.data_list[index]
        image_id = data['image_file']
        boxes = data['boxes']
        labels = data['labels']
        image = cv2.imread(image_id)
        for box, label in zip(boxes, labels):
            x0, y0, x1, y1 = [int(v) for v in box]
            color = (0, 0, 255) if label == 1 else (0, 255, 255)
            cv2.rectangle(image, (x0, y0), (x1, y1), color, 1)
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow("image", image)
        cv2.waitKey(-1)


if __name__ == '__main__':
    dataset = COCODataset("/media/test/data/coco")
    dataset.show_image_and_labels(7)
