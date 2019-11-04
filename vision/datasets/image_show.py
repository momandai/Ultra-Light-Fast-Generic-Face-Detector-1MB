from vision.datasets.voc_dataset import VOCDataset
import cv2

dataset = VOCDataset('/home/test/github/momandai/Ultra-Light-Fast-Generic-Face-Detector-1MB/data/wider_face_add_lm_10_10')

image_id = dataset.ids[2760]
image = dataset._read_image(image_id)
boxes, labels, is_difficult = dataset._get_annotation(image_id)

for box in boxes:
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255))

cv2.imshow("image", image)
cv2.waitKey(-1)