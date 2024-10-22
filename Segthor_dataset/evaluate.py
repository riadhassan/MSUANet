from copy import deepcopy
from multiprocessing.pool import Pool

from medpy import metric
import numpy as np

SMOOTH = 1e-6


### code is taken from
# https://github.com/himashi92/VT-UNet/blob/main/VTUNet/vtunet/evaluation/region_based_evaluation.py

def iou_numpy(outputs: np.array, labels: np.array):
    # outputs = outputs.squeeze(1)

    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    # thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10

    return iou.mean()  # Or thresholded.mean()


def get_Organ_regions():
    regions = {
        "esophagus": 1,
        "heart": 2,
        "trachea": 3,
        "aorta": 4
    }
    return regions


def get_LCTSC_regions():
    regions = {
        "Esophagus": 1,
        "Spinalcord": 2,
        "Heart": 3,
        "Left-lung": 4,
        "Right-lung": 5
    }
    return regions


def print_Thoracic(organ_dice, organ_hd):
    dice_dict = {"Esophegus Dice": organ_dice[0], "Heart": organ_dice[1], "Trachea": organ_dice[2],
                 "Aorta": organ_dice[3]}
    hd_dict = {"Esophegus HD": organ_hd[0], "Heart HD": organ_hd[1], "Trachea HD": organ_hd[2],
               "Aorta HD": organ_hd[3]}
    return dice_dict, hd_dict


def print_LCTSC(organ_dice, organ_hd):
    dice_dict = {"Esophegus Dice": organ_dice[0], "Spine Dice": organ_dice[1], "Heart Dice": organ_dice[2],
                 "Left Lung Dice": organ_dice[3], "Right Lung Dice": organ_dice[4]}
    hd_dict = {"Esophegus HD": organ_hd[0], "Spine HD": organ_hd[1], "Heart HD": organ_hd[2],
               "Left Lung HD": organ_hd[3], "Right Lung HD": organ_hd[4]}
    return dice_dict, hd_dict


def create_region_from_mask(mask, join_labels: tuple):
    mask_new = np.zeros_like(mask, dtype=np.uint8)
    mask_new[mask == join_labels] = 1
    '''
    for l in join_labels:
        mask_new[mask == l] = 1
    '''
    return mask_new


def evaluate_case(image_gt, image_pred, regions):
    # results = {}
    dice_values = []
    hd_values = []
    iou_values = []
    for r in regions:
        mask_pred = create_region_from_mask(image_pred, regions[r])
        mask_gt = create_region_from_mask(image_gt, regions[r])
        dc = np.nan if np.sum(mask_gt) == 0 and np.sum(mask_pred) == 0 else metric.dc(mask_pred, mask_gt)
        hd = np.nan if np.sum(mask_gt) == 0 or np.sum(mask_pred) == 0 else metric.asd(mask_pred, mask_gt)
        iou = np.nan if np.sum(mask_gt) == 0 or np.sum(mask_pred) == 0 else iou_numpy(mask_pred, mask_gt)
        dice_values.append(dc)
        hd_values.append(hd)
        iou_values.append(iou)

    # results["Dice"] = dice_values
    # results["HD"] = hd_values
    return dice_values, hd_values, iou_values