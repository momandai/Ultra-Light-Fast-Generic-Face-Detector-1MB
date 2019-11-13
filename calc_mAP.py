import argparse
import torch
import logging
import json
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

from vision.ssd.config.fd_config import define_img_size
from vision.datasets.coco_dataset import COCODataset
from torch.utils.data import DataLoader
define_img_size(640, "coco_person_face")

from vision.ssd.config import fd_config
from vision.ssd.data_preprocessing import TestTransform
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.ssd.ssd import MatchPrior
from vision.utils.box_utils import iou_of
config = fd_config

def calc_mAP(weights,
            batch_size=16,
            img_size=640,
            iou_thres=0.5,
            conf_thres=0.001,
            nms_thres=0.5,
            save_json=False,
            model=None):
    label_path = "models/train-coco_person_face-0.0.1-RFB/coco-person-face-labels.txt"
    class_names = [name.strip() for name in open(label_path).readlines()]
    num_classes = len(class_names)
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device="cuda:0")
    predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=100, device="cuda:0")

    net.load(weights)
    net.eval()
    device = "cuda:0"

    # target_transform = MatchPrior(config.priors, config.center_variance,
    #                               config.size_variance, 0.34999999404)
    #
    # test_transform = TestTransform(config.image_size, config.image_mean_test, config.image_std)
    # val_dataset = COCODataset("/media/test/data/coco", transform=test_transform,
    #                           target_transform=target_transform, is_test=True)
    # logging.info("validation dataset size: {}".format(len(val_dataset)))
    #
    # val_loader = DataLoader(val_dataset, batch_size,
    #                         num_workers=4,
    #                         shuffle=False)

    data_list = json.load(open("/media/test/data/coco/test_datalist.json"))

    all_correct = None
    all_p_prob = None
    all_p_label = None
    all_g_label = None
    seen = 0
    for data in tqdm(data_list):
        image_id = data['image_file']
        gt_boxes = np.array(data['boxes'], dtype=np.float32)
        gt_labels = np.array(data['labels'], dtype=np.int64)
        image = cv2.imread(image_id)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        p_boxes, p_labels, p_probs = predictor.predict(image)
        nl = gt_labels.shape[0]
        correct = np.array([0] * p_boxes.shape[0])
        for i, gt_box in enumerate(gt_boxes):
            seen += 1
            p_index = np.array(range(p_boxes.shape[0]))
            gt_label = gt_labels[i]
            valid_p_boxes = p_boxes[correct == 0]   # remove matched predic box
            valid_p_index = p_index[correct == 0]
            valid_p_probs = p_probs[correct == 0]
            valid_p_labels = p_labels[correct == 0]
            valid_p_boxes = valid_p_boxes[valid_p_labels == gt_label]  # select predict label == gt label
            valid_p_index = valid_p_index[valid_p_labels == gt_label]
            valid_p_probs = valid_p_probs[valid_p_labels == gt_label]
            if valid_p_boxes.shape[0] == 0:
                continue
            iou = iou_of(torch.tensor(valid_p_boxes), torch.tensor(np.expand_dims(gt_box, axis=0)))
            max_val = torch.max(iou)
            if max_val.item() > iou_thres:
                correct[valid_p_index[torch.argmax(iou).item()]] = 1
        all_correct = np.concatenate([all_correct, correct], axis=0) if all_correct is not None else correct
        all_p_prob = np.concatenate([all_p_prob, p_probs], axis=0) if all_p_prob is not None else p_probs
        all_p_label = np.concatenate([all_p_label, p_labels], axis=0) if all_p_label is not None else p_labels
        all_g_label = np.concatenate([all_g_label, gt_labels], axis=0) if all_g_label is not None else gt_labels
    p, r, ap, f1, ap_class = ap_per_class(all_correct, all_p_prob, all_p_label, all_g_label)
    mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
    nt = np.bincount(all_g_label.astype(np.int64), minlength=2)  # number of targets per class
    # Print results
    phead = '%30s' + '%10s' * 6
    print(phead % ('type', 'total', 'total', 'mp', 'mr', 'map', 'mf1'))
    pf = '%30s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    # Print results per class
    names = ['BACKGROUND', 'person', 'face']
    for i, c in enumerate(ap_class):
        print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))
pass



def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    l = [None, None]
    labels = ['BACKGROUND', 'person', 'face']
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

            # Plot
            plt.plot(recall_curve, precision_curve, label=labels[c])

    plt.legend()
    plt.show()
    plt.savefig("Ultra-fast-face-detect.png")
    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='calc_mAP.py')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--weights', type=str, default='models/train-coco_person_face-0.0.1-RFB/RFB-Epoch-199-Loss-3.3373507743790034.pth', help='weight path')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        mAP = calc_mAP(opt.weights,
                       opt.batch_size,
                       opt.img_size,
                       opt.iou_thres,
                       opt.conf_thres,
                       opt.nms_thres)

