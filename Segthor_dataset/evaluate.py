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


def print_Thoracic(organ_dice, organ_asd, organ_iou):
    dice_dict = {"Esophegus Dice": organ_dice[0], "Heart": organ_dice[1], "Trachea": organ_dice[2],
                 "Aorta": organ_dice[3]}
    asd_dict = {"Esophegus asd": organ_asd[0], "Heart asd": organ_asd[1], "Trachea asd": organ_asd[2],
               "Aorta asd": organ_asd[3]}
    iou_dict = {"Esophegus iou": organ_iou[0], "Heart iou": organ_iou[1], "Trachea iou": organ_iou[2],
                "Aorta iou": organ_iou[3]}
    return dice_dict, asd_dict, iou_dict


def print_LCTSC(organ_dice, organ_asd):
    dice_dict = {"Esophegus Dice": organ_dice[0], "Spine Dice": organ_dice[1], "Heart Dice": organ_dice[2],
                 "Left Lung Dice": organ_dice[3], "Right Lung Dice": organ_dice[4]}
    asd_dict = {"Esophegus asd": organ_asd[0], "Spine asd": organ_asd[1], "Heart asd": organ_asd[2],
               "Left Lung asd": organ_asd[3], "Right Lung asd": organ_asd[4]}
    return dice_dict, asd_dict


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
    asd_values = []
    iou_values = []
    for r in regions:
        mask_pred = create_region_from_mask(image_pred, regions[r])
        mask_gt = create_region_from_mask(image_gt, regions[r])
        dc = np.nan if np.sum(mask_gt) == 0 and np.sum(mask_pred) == 0 else metric.dc(mask_pred, mask_gt)
        asd = np.nan if np.sum(mask_gt) == 0 or np.sum(mask_pred) == 0 else metric.asd(mask_pred, mask_gt)
        iou = np.nan if np.sum(mask_gt) == 0 or np.sum(mask_pred) == 0 else iou_numpy(mask_pred, mask_gt)
        dice_values.append(dc)
        asd_values.append(asd)
        iou_values.append(iou)

    # results["Dice"] = dice_values
    # results["asd"] = asd_values
    return dice_values, asd_values, iou_values