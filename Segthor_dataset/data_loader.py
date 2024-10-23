import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import glob
from scipy.io import loadmat
from Segthor_dataset.data_aug import augment_image_label


class OrganSegmentationDataset(Dataset):
    def __init__(
            self,
            images_dir,
            subset,
            train_st=35,
            test_st=5
    ):
        if subset == 'train':
            self.images_dir = os.path.join(images_dir, "train")
        else:
            self.images_dir = os.path.join(images_dir, "test")

        self.subset = subset
        self.traning_patient = train_st
        self.test_patient = test_st
        self.data_paths = []
        self.patient_ids = []
        self.required_test = False

        print("reading {} images...".format(subset))
        filesPath = glob.glob(self.images_dir + os.sep+ "*.mat")

        if (subset == "train"):
            for filePath in sorted(filesPath):
                patient_id = int(filePath.split(os.sep)[-1].split("_")[1])
                # if patient_id <= self.traning_patient:
                #  if patient_id not in self.patient_ids:
                self.patient_ids.append(patient_id)
                self.data_paths.append(filePath)
                self.data_paths = sorted(self.data_paths)
            self.patient_ids = np.unique(np.array(self.patient_ids))

        elif (subset == "test"):
            ids = [os.path.basename(x).split('.')[0].split('_')[1] for x in filesPath]
            ids = np.unique(np.array(ids))
            self.val_patients = {}
            for id in ids:
                pattern = f"Patient_{id}_S_*.mat"
                slices = glob.glob(os.path.join(self.images_dir, pattern))
                patient = f"Patient_{id}"
                slices.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
                self.val_patients[patient] = slices
            '''
            filesPath = (filesPath)
            for filePath in sorted(filesPath):
              patient_id = int(filePath.split("/")[-1].split("_")[1])
              if self.traning_patient+self.test_patient >= patient_id > self.traning_patient:
                if patient_id not in self.patient_ids:
                  self.patient_ids.append(patient_id)
                self.data_paths.append(filePath)
                self.data_paths = sorted(self.data_paths)
            '''
        # print(self.patient_ids)

    def normalize_data(self, data):
        # data[data< -150]=-150
        # data[data>200]= 200 #1524
        # data=data +150
        # data=(data*2.)/200 - 1 #1500 - 1
        data = data / data.max()
        return (data)

    def __len__(self):
        if self.subset == "train":
            return len(self.data_paths)
        elif self.subset == "test":
            return len(self.val_patients)

    def __getitem__(self, id):

        if self.subset == "train":
            filePath = self.data_paths[id]
            file_name = filePath.split(os.sep)[-1].split("_")
            patient_id = int(file_name[1])
            slice_id = int(file_name[3].split(".")[0])
            mat = loadmat(filePath)
            mask = mat['mask']
            image = mat['img']
            mask = mask[None, :, :]
            image = image[None, :, :]

            image, mask = augment_image_label(np.squeeze(image), np.squeeze(mask), 256,
                                              trans_threshold=.6, horizontal_flip=None, rotation_range=30,
                                              height_shift_range=0.05, width_shift_range=0.05,
                                              shear_range=None, zoom_range=(0.95, 1.05), elastic=None, add_noise=0.0)
            image = self.normalize_data(image)
            image_tensor = torch.from_numpy(image.astype(np.float32))
            mask_tensor = torch.from_numpy(mask.astype(int))
            return image_tensor, mask_tensor, patient_id, slice_id

        elif self.subset == "test":
            key = list(self.val_patients.keys())[id]
            patient = self.val_patients[key]
            return patient, key


def data_loaders(data_dir):
    dataset_train, dataset_valid = datasets(data_dir)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=6,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid


def datasets(data_dir):
    train = OrganSegmentationDataset(images_dir=data_dir,
                                     subset="train"
                                     )
    valid = OrganSegmentationDataset(images_dir=data_dir,
                                     subset="test"
                                     )
    return train, valid