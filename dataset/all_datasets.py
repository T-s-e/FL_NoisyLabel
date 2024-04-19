import os
import numpy as np
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset



class isic2019(Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode
        assert self.mode in ["train", "test"]
        self.transform = transform

        csv_file = os.path.join(self.root, self.mode+".csv")
        self.file = pd.read_csv(csv_file)
        self.images = self.file["image"].values
        self.labels = self.file.iloc[:, 1:].values.astype("int")
        self.targets = np.argmax(self.labels, axis=1)
        self.n_classes = len(np.unique(self.targets))
        assert self.n_classes == 8

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(
            self.root, "ISIC_2019_Training_Input", self.images[index]+".jpg")
        img = Image.open(image_path).convert("RGB")
        img = self.transform(img)
        label = self.targets[index]
        return img, label

    

class ICH(Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode
        assert self.mode in ["train", "test"]
        self.transform = transform

        csv_file = os.path.join(self.root, self.mode+".csv")
        self.file = pd.read_csv(csv_file)
        self.images = self.file["id"].values
        self.labels = self.file.iloc[:, 1:].values.astype("int")
        self.targets = np.argmax(self.labels, axis=1)
        self.n_classes = len(np.unique(self.targets))
        assert self.n_classes == 5

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        id, target = self.images[index], self.targets[index]
        img = self.read_image(id)

        img = self.transform(img)
        return img, target

    def read_image(self, id):
        image_path = os.path.join(self.root, "stage_1_train_images", id+".png")
        image = Image.open(image_path).convert("RGB")
        return image


class Camelyon17(Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode
        assert self.mode in ["train", "test"]
        self.transform = transform

        # csv_file = os.path.join(self.root, self.mode+".csv")
        csv_file = os.path.join(self.root, self.mode+"_staining.csv")
        self.file = pd.read_csv(csv_file)
        self.images = self.file["id"].values                # relative path: patches/patient_017_node_4/patch_patient_017_node_4_x_1664_y_26336.png
        self.targets = self.file["class"].values.astype("int")
        self.n_classes = len(np.unique(self.targets))
        # add new attributes of "staining" and "s_classes"
        self.staining = self.file['staining'].values.astype("int")
        self.s_classes = len(np.unique(self.staining))


        print('self.n_classes', self.n_classes)
        assert self.n_classes == 2

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        id, target = self.images[index], self.targets[index]
        img = self.read_image(id)

        img = self.transform(img)
        return img, target

    def read_image(self, id):
        image_path = os.path.join(self.root, id)
        image = Image.open(image_path).convert("RGB")
        return image
