import logging
import os
import pathlib
import xml.etree.ElementTree as ET
import json

import cv2
import numpy as np


class VOCPersonDataset:

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
            image_sets_file = self.root / "ImageSets/Main/test.txt"
            self.datalist_name = "test_data_list.json"

        else:
            image_sets_file = self.root / "ImageSets/Main/trainval.txt"
            self.datalist_name = "train_data_list.json"
        self.ids = VOCPersonDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult
        self.datalist = []

        logging.info("No labels file, using default VOC classes.")
        self.class_names = ('BACKGROUND',
                            'person')

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

        for image_id in self.ids:
            image_file = self.root / f"JPEGImages/{image_id}.jpg"
            if not os.path.exists(image_file):
                print("{0} is not exist!".format(image_file))
                continue
            boxes, labels, is_difficult = self._get_annotation(image_id)
            if boxes is None or labels is None:
                # print("none")
                continue
            if not self.keep_difficult:
                boxes = boxes[is_difficult == 0]
                labels = labels[is_difficult == 0]
            if boxes.shape[0] == 0:
                # print("no bbox! {0}".format(image_id))
                continue
            self.datalist.append({"image_id": image_id, "boxes": boxes, "labels": labels})

        print("dataset laoded, length: {0}".format(len(self.datalist)))

    def __getitem__(self, index):
        data = self.datalist[index]
        image_id = data["image_id"]
        boxes = data["boxes"]
        labels = data["labels"]
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        # print(image.shape, flush=True)
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.datalist)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = self.root / f"Annotations/{image_id}.xml"
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            # we're only concerned with clases in our list
            if class_name in self.class_dict:
                bbox = object.find('bndbox')

                # VOC dataset format follows Matlab, in which indexes start from 0
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                boxes.append([x1, y1, x2, y2])

                labels.append(self.class_dict[class_name])
                is_difficult_str = object.find('difficult').text
                is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        if not boxes:
            return None, None, None

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_id):
        image_file = self.root / f"JPEGImages/{image_id}.jpg"
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def show_image_and_label(self, index):
        data = self.datalist[index]
        image_id = data["image_id"]
        boxes = data["boxes"]
        labels = data["labels"]
        image = self._read_image(image_id)
        for box in boxes:
            x0, y0, x1, y1 = [int(v) for v in box]
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 1)
        cv2.imshow("image", image)
        cv2.imwrite("test.jpg", image)
        cv2.waitKey(-1)


if __name__ == '__main__':
    a = np.array([[1,2,3,4]])
    dataset = VOCPersonDataset("/media/test/data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012")
    dataset.show_image_and_label(238)